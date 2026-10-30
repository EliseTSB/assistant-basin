"""Tests for tools/part_check.py and tools/launch_arc.py, on throwaway git repos. Written by Claude (Opus 5.5),
2026-10-29.

Run: python -m pytest -q tools/test_tools.py. PART_CHECK_TOOLS=<dir> tests a copy of the tools instead, which is how
the deliberate breaks below are run (mutate a copy, never the tracked file):
  1. part_check.py: make VERDICT match anything (re.compile(r".")) -> test_publish_missing_verdict fails.
  2. part_check.py: drop the 'Measured:' check for runs over 5 minutes -> test_prelaunch_long_run_needs_measurement fails.
  3. launch_arc.py: make job_busy return (False, "") always -> test_job_busy fails.
"""
import json, os, subprocess, sys
from pathlib import Path

import pytest

TOOLS = Path(os.environ.get("PART_CHECK_TOOLS", Path(__file__).resolve().parent))
sys.path.insert(0, str(TOOLS))
import part_check as pc  # noqa: E402
import launch_arc as la  # noqa: E402

DESIGN = """# Part 900: does a complete design pass?

Procedure: v1
Registered 2026-10-29 19:00 CT by Claude (Opus 5.5), before any data. Asked by the test suite.
The commit that adds this file is the registration.

## Question

Whether the checker accepts a complete design.

## Prior art and existing data

- Closest published work: none found: a fixture.
- Our own data that already bears on it: none found.

## Method

A fixture; nothing runs.

## Controls

Matched random controls: 10 directions per layer at the same written norm.

Gate: the do-nothing hook reproduces the plain model bit for bit.

## Size

- Smallest effect worth knowing: 30% against 5%, which needs n = 29 per arm.
- Total: 1,200 generations, about 20 minutes on the A770.
- Pilot: 64 generations; stop if 0 of 32 move either way.

## Optimization

- Bound resource: GPU
- Measured: 2.0 answers/s at batch 64 on 64 pilot rows
- Levers: batch 96 taken (2.6 answers/s); shared-prompt prefill declined (the prompts differ).

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 900.1 | The checker passes. | high | Claude |
| 900.2 | Nothing fails. | high | Claude |

## Reading rules

score900.py counts passes; error rows are dropped and counted.

## Limits

A fixture.
"""

PREDICTIONS = """# Predictions

## Part 899 (registered 2026-10-28, PART899-DESIGN.md): another part
| 899.1 | x | | |

## Part 900 (registered 2026-10-29, PART900-DESIGN.md): does a complete design pass?
| 900.1 | Claude (high): The checker passes. | | |
| 900.2 | Claude (high): Nothing fails. | | |

## Part 900b (registered 2026-10-29, PART900B-DESIGN.md): a sibling part
| 900b.1 | x | | |
"""

README = """# Part 900: the checker passes a complete design

Two predictions, both held.

## Predictions

| # | Prediction | Result | Verdict |
| --- | --- | --- | --- |
| 900.1 | The checker passes. | rc 0 | **held** |
| 900.2 | Nothing fails. | 0 failures | **held** |

## Files

- `results/part900/score.json`: the counts; made by `part900/score900.py`.

Cold check: Haiku 5.5, 2026-10-29, 0 findings.
"""


def sh(*args, cwd):
    return subprocess.run(list(args), cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    origin, r = tmp_path / "origin.git", tmp_path / "repo"
    sh("git", "init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    sh("git", "init", "-q", "-b", "main", str(r), cwd=tmp_path)
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        sh("git", "config", k, v, cwd=r)
    sh("git", "remote", "add", "origin", str(origin), cwd=r)
    (r / "PART900-DESIGN.md").write_text(DESIGN, encoding="utf-8")
    (r / "PREDICTIONS.md").write_text(PREDICTIONS, encoding="utf-8")
    (r / "part900").mkdir()
    (r / "part900" / "score900.py").write_text("print(1)\n", encoding="utf-8")
    sh("git", "add", "-A", cwd=r)
    sh("git", "commit", "-q", "-m", "register 900", cwd=r)
    sh("git", "push", "-q", "origin", "main", cwd=r)
    return r


def write_results(r, readme=README, commit=True):
    d = r / "results" / "part900"
    d.mkdir(parents=True, exist_ok=True)
    (d / "README.md").write_text(readme, encoding="utf-8")
    (d / "score.json").write_text("{}\n", encoding="utf-8")
    if commit:
        sh("git", "add", "-A", cwd=r)
        sh("git", "commit", "-q", "-m", "results", cwd=r)


def edit(r, old, new, name="PART900-DESIGN.md", push=False):
    p = r / name
    t = p.read_text(encoding="utf-8")
    assert old in t, old
    p.write_text(t.replace(old, new), encoding="utf-8")
    if push:
        sh("git", "commit", "-q", "-am", "amend", cwd=r)
        sh("git", "push", "-q", "origin", "main", cwd=r)


def test_prelaunch_complete_design_passes(repo):
    rep = pc.prelaunch(repo, "900", fetch=False)
    assert rep.failed == [], rep.failed


def test_prelaunch_long_run_needs_measurement(repo):
    edit(repo, "Measured: 2.0 answers/s at batch 64 on 64 pilot rows", "Measured: n/a: under 5 minutes")
    assert any("Measured" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_prelaunch_short_run_needs_no_measurement(repo):
    edit(repo, "about 20 minutes", "about 3 minutes")
    edit(repo, "Total: 1,200 generations", "Total: 900 generations")
    edit(repo, "Measured: 2.0 answers/s at batch 64 on 64 pilot rows", "Measured: n/a: under 5 minutes", push=True)
    assert pc.prelaunch(repo, "900", fetch=False).failed == []


def test_prelaunch_big_run_needs_a_pilot(repo):
    edit(repo, "Pilot: 64 generations; stop if 0 of 32 move either way.", "Pilot: not needed: small.")
    assert any("pilot" in f.lower() for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_prelaunch_placeholder_time_fails(repo):
    edit(repo, "Asked by the test suite.", "Asked by the test suite at 18:4x.")
    assert any("placeholder" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_prelaunch_unfilled_template_fails(repo):
    edit(repo, "A fixture; nothing runs.", "{{Model and pinned revision}}")
    assert any("template" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_prelaunch_controls_need_random_and_gate(repo):
    edit(repo, "Matched random controls: 10 directions per layer at the same written norm.", "Some controls.")
    edit(repo, "Gate: the do-nothing hook reproduces the plain model bit for bit.", "Nothing else.")
    failed = pc.prelaunch(repo, "900", fetch=False).failed
    assert any("random" in f for f in failed) and any("Gate" in f for f in failed)


def test_prelaunch_unregistered_part_fails(repo):
    (repo / "PART901-DESIGN.md").write_text(DESIGN.replace("900", "901"), encoding="utf-8")
    edit(repo, "## Part 900b", "## Part 901 (registered 2026-10-29, PART901-DESIGN.md): local only\n"
         "| 901.1 | x | | |\n| 901.2 | y | | |\n\n## Part 900b", name="PREDICTIONS.md")
    failed = pc.prelaunch(repo, "901", fetch=False).failed
    assert any("not registered" in f for f in failed), failed


def test_prelaunch_legacy_design_needs_flag(repo):
    edit(repo, "Procedure: v1\n", "")
    assert pc.prelaunch(repo, "900", fetch=False).failed
    assert pc.prelaunch(repo, "900", fetch=False, legacy_ok=True).failed == []


def test_predictions_section_excludes_sibling_part():
    sec = pc.predictions_section(PREDICTIONS, "900")
    assert pc.pred_ids(sec, "900") == ["900.1", "900.2"]
    assert "900b.1" not in sec


def test_publish_passes(repo):
    write_results(repo)
    rep = pc.publish(repo, "900")
    assert rep.failed == [], rep.failed


def test_publish_missing_verdict(repo):
    write_results(repo, README.replace("| 900.2 | Nothing fails. | 0 failures | **held** |", "| 900.2 | Nothing fails. | 0 | |"))
    assert any("900.2" in f for f in pc.publish(repo, "900").failed)


def test_publish_needs_cold_check_line(repo):
    write_results(repo, README.replace("Cold check: Haiku 5.5, 2026-10-29, 0 findings.\n", ""))
    assert any("Cold check" in f for f in pc.publish(repo, "900").failed)


def test_publish_missing_and_local_paths(repo):
    write_results(repo, README + "\n- `results/part900/gone.json` and /home/someone/run.log\n")
    failed = pc.publish(repo, "900").failed
    assert any("missing" in f and "gone.json" in f for f in failed)
    assert any("local or out-of-repo" in f for f in failed)


def test_publish_uncommitted_path(repo):
    write_results(repo)
    (repo / "results" / "part900" / "new.json").write_text("{}", encoding="utf-8")
    edit(repo, "Cold check:", "- `results/part900/new.json`: new.\n\nCold check:", name="results/part900/README.md")
    assert any("not committed" in f for f in pc.publish(repo, "900").failed)


def test_job_busy():
    assert la.job_busy("10:00:00 step 1 start: x\r\n10:00:05 step 1 gpu: y\r\n")[0] is True
    assert la.job_busy("10:00:00 step 1 start: x\n10:10:00 step 1 rc=0\n")[0] is False
    assert la.job_busy("10:00:00 step 1 start: x\n10:10:00 step 1 rc=0\n10:11:00 step 1 start: z\n")[0] is True
    assert la.job_busy("10:00:00 REFUSED: another job.py holds job.lock\n")[0] is False


def test_write_steps_crlf(tmp_path):
    la.write_steps(tmp_path, ["-I -B a.py", "-I -B b.py --x 1"])
    assert (tmp_path / "steps.txt").read_bytes() == b"-I -B a.py\r\n-I -B b.py --x 1\r\n"


def test_cpu_script_has_no_local_path():
    s = la.cpu_script("/opt/venv/bin/python -I -B part900/score900.py")
    assert 'cd "$(dirname "$0")/../.."' in s and "home" not in s


def fake_job(tmp_path, line):
    job = tmp_path / "job"
    job.mkdir()
    (job / "progress.log").write_text("09:00:00 step 1 start: old\n09:30:00 step 1 rc=0\n", encoding="utf-8")
    fake = tmp_path / "schtasks"
    fake.write_text(f"#!/bin/bash\nprintf '%s\\r\\n' '{line}' >> {job}/progress.log\necho SUCCESS\n", encoding="utf-8")
    fake.chmod(0o755)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"arc_job_dir": str(job), "schtasks": str(fake), "task_name": "T"}), encoding="utf-8")
    return job, {"LAUNCH_ARC_CONFIG": str(cfg), "PATH": os.environ.get("PATH", "")}


def test_launch_refuses_failed_check(repo, tmp_path):
    job, env = fake_job(tmp_path, "10:00:00 step 1 start: new")
    edit(repo, "Measured: 2.0 answers/s at batch 64 on 64 pilot rows", "Measured: n/a")
    rc = la.main(["900", "--step", "-I -B x.py", "--repo", str(repo), "--no-fetch", "--wait", "3"], env=env)
    assert rc == 1 and not (job / "steps.txt").exists()


def test_launch_starts_and_records_baseline(repo, tmp_path):
    job, env = fake_job(tmp_path, "10:00:00 step 1 start: -I -B part900\\score900.py")
    rc = la.main(["900", "--step", "-I -B part900\\score900.py", "--copy", "part900/score900.py", "--repo", str(repo),
                  "--no-fetch", "--wait", "5"], env=env)
    assert rc == 0
    assert (job / "part900" / "score900.py").exists()
    rec = json.loads((repo / "results" / "part900" / "launch.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec["baseline_lines"] == 2 and rec["first_line"].endswith("score900.py")


def test_launch_override_is_recorded(repo, tmp_path):
    job, env = fake_job(tmp_path, "10:00:00 step 1 start: -I -B y.py")
    edit(repo, "Measured: 2.0 answers/s at batch 64 on 64 pilot rows", "Measured: n/a")
    rc = la.main(["900", "--step", "-I -B y.py", "--repo", str(repo), "--no-fetch", "--wait", "5",
                  "--override", "rerun of a measured job"], env=env)
    rec = json.loads((repo / "results" / "part900" / "launch.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rc == 0 and rec["override"] == "rerun of a measured job" and rec["check_failed"]


def test_launch_refuses_busy_job(repo, tmp_path):
    job, env = fake_job(tmp_path, "x")
    (job / "progress.log").write_text("09:00:00 step 1 start: still going\n", encoding="utf-8")
    rc = la.main(["900", "--step", "-I -B y.py", "--repo", str(repo), "--no-fetch", "--wait", "3"], env=env)
    assert rc == 1 and not (job / "steps.txt").exists()


# Round-1 review findings (cheap-review procedure-v1, DeepSeek V4.1 Flash, 2026-10-29), one test each.

def test_publish_empty_verdict_cells_fail(repo):     # F1
    bad = README.replace("| 900.1 | The checker passes. | rc 0 | **held** |", "| 900.1 | The pattern held at 0.8. | 0.81 | |")
    write_results(repo, bad)
    assert any("900.1" in f for f in pc.publish(repo, "900").failed)


def test_prelaunch_prefix_id_is_not_registration(repo):    # F2
    edit(repo, "| 900.1 | Claude (high): The checker passes. | | |", "| 900.10 | Claude (high): The checker passes. | | |",
         name="PREDICTIONS.md", push=True)
    assert any("900.1" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_prelaunch_amended_design_fails_until_registered(repo):    # F3
    edit(repo, "about 20 minutes", "about 900 minutes")
    assert any("differs from its registered copy" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)
    sh("git", "commit", "-q", "-am", "amend", cwd=repo)
    sh("git", "push", "-q", "origin", "main", cwd=repo)
    assert not any("differs" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_prelaunch_design_registered_but_not_predictions(repo):    # F15
    edit(repo, "## Part 900b", "## Part 901 (registered 2026-10-29, PART901-DESIGN.md): x\n| 901.1 | x | | |\n| 901.2 | y | | |\n\n## Part 900b",
         name="PREDICTIONS.md")
    (repo / "PART901-DESIGN.md").write_text(DESIGN.replace("900", "901"), encoding="utf-8")
    sh("git", "add", "PART901-DESIGN.md", cwd=repo)
    sh("git", "commit", "-q", "-m", "design only", "--", "PART901-DESIGN.md", cwd=repo)
    sh("git", "push", "-q", "origin", "main", cwd=repo)
    failed = pc.prelaunch(repo, "901", fetch=False).failed
    assert any("PREDICTIONS.md lacks" in f for f in failed) and not any("is not on origin/main" in f for f in failed)


def test_prelaunch_predictions_registered_but_not_design(repo):    # F15
    edit(repo, "## Part 900b", "## Part 901 (registered 2026-10-29, PART901-DESIGN.md): x\n| 901.1 | x | | |\n| 901.2 | y | | |\n\n## Part 900b",
         name="PREDICTIONS.md", push=True)
    (repo / "PART901-DESIGN.md").write_text(DESIGN.replace("900", "901"), encoding="utf-8")
    failed = pc.prelaunch(repo, "901", fetch=False).failed
    assert any("is not on origin/main" in f for f in failed) and not any("lacks" in f for f in failed)


def test_prelaunch_missing_section(repo):    # F15
    edit(repo, "## Limits\n\nA fixture.\n", "", push=True)
    assert any("Limits" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_prelaunch_fetch_failure_is_reported(repo, tmp_path):    # F15
    import shutil
    shutil.rmtree(tmp_path / "origin.git")
    assert any("git fetch" in f for f in pc.prelaunch(repo, "900", fetch=True).failed)


def test_publish_without_predictions_file_does_not_crash(repo):    # F6
    write_results(repo)
    (repo / "PREDICTIONS.md").unlink()
    assert pc.publish(repo, "900").failed


def test_publish_out_of_repo_paths_fail_cleanly(repo):    # F6
    write_results(repo, README + "\n- `/etc/hosts` and `../outside/x.json`\n")
    assert any("out-of-repo" in f for f in pc.publish(repo, "900").failed)


def test_publish_accepts_mixed_and_not_tested(repo):    # F7
    write_results(repo, README.replace("| rc 0 | **held** |", "| rc 0 | mixed |").replace("| 0 failures | **held** |", "| - | not tested |"))
    assert pc.publish(repo, "900").failed == []


def test_publish_checks_paths_inside_commands(repo):    # F8
    write_results(repo, README + "\nMade by `python part900/gone.py --out results/part900/x.json`.\n")
    assert any("part900/gone.py" in f for f in pc.publish(repo, "900").failed)


def test_sections_ignore_fenced_headings():    # F9
    t = "## Size\n\n```\n## inputs\n1,200 rows\n```\n\n- Total: 1,200 generations, about 20 minutes\n\n## Limits\nx\n"
    sec = pc.sections(t)
    assert "inputs" not in sec and "Total:" in sec["Size"]


def test_prelaunch_measured_needs_a_rate(repo):    # F13
    edit(repo, "Measured: 2.0 answers/s at batch 64 on 64 pilot rows", "Measured: not measured yet; the plan is batch 64",
         push=True)
    assert any("Measured" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_parse_size_million_and_zero():    # F14
    assert pc.parse_size("- Total: 1.5 million generations, about 20 minutes")[0] == 1500000
    assert pc.parse_size("- Total: 0 generations, about 0 minutes")[0] == 0


def test_prelaunch_zero_size_fails(repo):    # F14
    edit(repo, "Total: 1,200 generations, about 20 minutes", "Total: 0 generations, about 0 minutes", push=True)
    assert any("above zero" in f for f in pc.prelaunch(repo, "900", fetch=False).failed)


def test_launch_lock_refuses_a_second_launch(repo, tmp_path):    # F4
    job, env = fake_job(tmp_path, "10:00:00 step 1 start: -I -B y.py")
    (job / "launch.lock").write_text("123 10:00:00\n", encoding="utf-8")
    rc = la.main(["900", "--step", "-I -B y.py", "--repo", str(repo), "--no-fetch", "--wait", "3"], env=env)
    assert rc == 1 and not (job / "steps.txt").exists()


def test_launch_ignores_another_launchs_start_line(repo, tmp_path):    # F4
    job, env = fake_job(tmp_path, "10:00:00 step 1 start: -I -B SECOND.py")
    rc = la.main(["900", "--step", "-I -B FIRST.py", "--repo", str(repo), "--no-fetch", "--wait", "3"], env=env)
    assert rc == 3 and not (job / "launch.lock").exists()


def test_launch_survives_a_truncated_log(repo, tmp_path):    # F5
    job, env = fake_job(tmp_path, "x")
    fake = tmp_path / "schtasks"
    fake.write_text(f"#!/bin/bash\nprintf '%s\\r\\n' '10:00:00 step 1 start: -I -B y.py' > {job}/progress.log\necho SUCCESS\n",
                    encoding="utf-8")
    rc = la.main(["900", "--step", "-I -B y.py", "--repo", str(repo), "--no-fetch", "--wait", "5"], env=env)
    rec = json.loads((repo / "results" / "part900" / "launch.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rc == 0 and rec["progress_log_shrank"] is True


def test_launch_copy_outside_repo_refused(repo, tmp_path):    # F10
    job, env = fake_job(tmp_path, "x")
    (tmp_path / "escaped.txt").write_text("x", encoding="utf-8")
    rc = la.main(["900", "--step", "-I -B y.py", "--copy", "../escaped.txt", "--repo", str(repo), "--no-fetch"], env=env)
    assert rc == 2 and not (job / "steps.txt").exists()


def test_launch_wait_zero_does_not_wait(repo, tmp_path):    # F11
    job, env = fake_job(tmp_path, "10:00:00 step 1 start: -I -B y.py")
    assert la.main(["900", "--step", "-I -B y.py", "--repo", str(repo), "--no-fetch", "--wait", "0"], env=env) == 0


def test_launch_cpu_with_copy_refused(repo, tmp_path):    # F12
    rc = la.main(["900", "--cpu", "true", "--copy", "part900/score900.py", "--repo", str(repo), "--no-fetch", "--dry-run"])
    assert rc == 2


def test_publish_pending_cold_check_fails(repo):
    write_results(repo, README.replace("Cold check: Haiku 5.5, 2026-10-29, 0 findings.", "Cold check: pending."))
    assert any("Cold check" in f for f in pc.publish(repo, "900").failed)


def _reg_repo(tmp_path):
    """A working clone of a bare remote whose PREDICTIONS.md has Parts 1 and 2."""
    remote, work = tmp_path / "remote.git", tmp_path / "work"
    sh("git", "init", "-q", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    sh("git", "clone", "-q", str(remote), str(work), cwd=tmp_path)
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        sh("git", "config", k, v, cwd=work)
    (work / "PREDICTIONS.md").write_text("# P\n\n## Part 1 (registered x): a\n\n| 1.1 | p | | |\n\n## Part 2 (registered x): b\n\n| 2.1 | q | | |\n", encoding="utf-8")
    (work / "D2.md").write_text("design 2\n", encoding="utf-8")
    sh("git", "add", ".", cwd=work)
    sh("git", "commit", "-q", "-m", "base", cwd=work)
    sh("git", "push", "-q", "origin", "HEAD:main", cwd=work)
    return remote, work


def _register(work, part, *files):
    tool = Path(__file__).resolve().parent / "register_public.py"
    return subprocess.run([sys.executable, str(tool), "--part", part, "--message", f"register {part}", *files],
                          cwd=work, capture_output=True, text=True)


def test_register_amendment_of_an_older_part_leaves_public_predictions_alone(tmp_path):
    remote, work = _reg_repo(tmp_path)
    p = work / "PREDICTIONS.md"
    p.write_text(p.read_text(encoding="utf-8").replace("| 1.1 | p | | |", "| 1.1 | p | 0.4 | held |"), encoding="utf-8")
    (work / "D2.md").write_text("design 2\n\n## Amendment 1\n", encoding="utf-8")
    r = _register(work, "2", "D2.md")
    assert r.returncode == 0, r.stderr
    pub = sh("git", "show", "origin/main:PREDICTIONS.md", cwd=work).stdout
    assert "| 1.1 | p | | |" in pub and pub.count("## Part 2 (registered") == 1
    assert "Amendment 1" in sh("git", "show", "origin/main:D2.md", cwd=work).stdout
    assert "| 1.1 | p | 0.4 | held |" in p.read_text(encoding="utf-8")   # the local verdicts survive the merge-back
    assert "Amendment 1" in (work / "D2.md").read_text(encoding="utf-8")


def test_register_new_part_publishes_only_its_own_section(tmp_path):
    remote, work = _reg_repo(tmp_path)
    p = work / "PREDICTIONS.md"
    s = p.read_text(encoding="utf-8").replace("| 2.1 | q | | |", "| 2.1 | q | 0.9 | failed |")
    p.write_text(s + "\n## Part 3 (registered y): c\n\n| 3.1 | r | | |\n", encoding="utf-8")
    (work / "D3.md").write_text("design 3\n", encoding="utf-8")
    r = _register(work, "3", "D3.md")
    assert r.returncode == 0, r.stderr
    pub = sh("git", "show", "origin/main:PREDICTIONS.md", cwd=work).stdout
    assert "| 3.1 | r | | |" in pub and "0.9 | failed" not in pub
    assert (work / "D3.md").exists() and "0.9 | failed" in p.read_text(encoding="utf-8")



def test_cited_paths_ignores_a_quoted_answers_line_break_slashes():
    import part_check as pc
    text = "Quoted: `I can assist with: / / - **Aristoxim**` and the file `results/part117/score.json`."
    assert sorted(pc.cited_paths(text)) == ["results/part117/score.json"]
