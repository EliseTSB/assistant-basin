"""launch_arc.uncopied_local_imports: refuse a copied file's import of a repo module that is neither copied nor in the
job folder; warn when the job folder's copy differs from the repo. Written 2026-11-19 after Part 212's pilot was
launched without score212.py, which it imports inside its stopping rule at the run's last step. Break: make the function
return ([], []) (the refuse and warn cases go red), drop the function-level walk (the lazy-import case goes red), or
compare copies by existence only (the stale-copy case goes red)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch_arc as la  # noqa: E402


def _w(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_a_lazy_import_of_an_uncopied_repo_module_is_refused(tmp_path):
    repo, job = tmp_path / "repo", tmp_path / "job"
    _w(repo, "part9/run9.py", "import json\n\ndef rule():\n    import score9\n    return score9.x\n")
    _w(repo, "part9/score9.py", "x = 1\n")
    job.mkdir()
    refuse, warn = la.uncopied_local_imports(repo, ["part9/run9.py"], job)
    assert refuse == [("part9/run9.py", 4, "score9", "part9/score9.py")] and warn == []
    assert la.uncopied_local_imports(repo, ["part9/run9.py", "part9/score9.py"], job) == ([], [])


def test_root_and_package_layouts_and_identical_job_copies(tmp_path):
    repo, job = tmp_path / "repo", tmp_path / "job"
    _w(repo, "part66/part66.py", "import part25\nimport circuit\nimport numpy\n")
    _w(repo, "part25.py", "a = 1\n")
    _w(repo, "circuit.py", "b = 2\n")
    _w(job, "circuit.py", "b = 2\n")
    refuse, warn = la.uncopied_local_imports(repo, ["part66/part66.py"], job)
    assert refuse == [("part66/part66.py", 1, "part25", "part25.py")] and warn == []


def test_a_stale_job_copy_warns_and_the_optional_list_is_skipped(tmp_path):
    repo, job = tmp_path / "repo", tmp_path / "job"
    _w(repo, "run.py", "import helper\ndef judge():\n    from judge_client import call_prompt\n")
    _w(repo, "helper.py", "v = 2\n")
    _w(repo, "judge_client.py", "def call_prompt(): pass\n")
    _w(job, "helper.py", "v = 1\n")
    refuse, warn = la.uncopied_local_imports(repo, ["run.py"], job)
    assert refuse == [] and warn == [("run.py", 1, "helper", "helper.py")]


def test_part_212s_real_pilot_copy_list(tmp_path):
    repo = HERE.parent
    if not (repo / "part212" / "run212.py").is_file():
        return
    copies = ["part212/run212.py", "part212/prompts212.json", "neuron_report.py", "part66/part66.py", "part25.py",
              "circuit.py"]
    refuse, _ = la.uncopied_local_imports(repo, copies, tmp_path)
    assert ("score212" in {m for _, _, m, _ in refuse})
    refuse, _ = la.uncopied_local_imports(repo, copies + ["part212/score212.py"], tmp_path)
    assert refuse == []
