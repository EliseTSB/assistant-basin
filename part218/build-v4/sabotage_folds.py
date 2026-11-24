import shutil, subprocess, sys, tempfile
from pathlib import Path
SRC = Path('/home/elisepc/green/worktrees/ab-part218/part218')
PY = '/home/elisepc/dspy-wsp/.venv/bin/python'
cases = [
 ('run218.py', "            if isinstance(record, dict) and 'item_index' not in record:\n                legacy += 1\n                continue\n", "", 'test_run218.py::test_read_answers_refuses_pre_v4_rows_but_drops_a_torn_line'),
 ('run218.py', "default='results218-v4'", "default='results218'", 'test_run218.py::test_default_out_is_a_v4_root_not_the_v3_pilot'),
 ('run218.py', "        if previous_gate.get('probe_ran'):", "        if False:", 'test_run218.py::test_tiny_model_end_to_end_smoke_pilot_gate_and_full'),
 ('score218.py', "    _refuse_pre_v4(out)\n", "", 'test_score218.py::test_the_scorer_refuses_a_pre_v4_run_dir'),
 ('score218.py', "        if len(rows) < n_expected:\n            incomplete += 1", "        if False:\n            incomplete += 1", 'test_score218.py::test_zero_row_identity_counts_a_short_item_as_incomplete'),
 ('score218.py', "    if status in ('running', 'incomplete'):", "    if False:", 'test_score218.py::test_a_run_that_did_not_finish_is_not_read'),
 ('score218.py', "        if push.get('no_nonzero_push') is not True:", "        if False:", 'test_score218.py::test_pilot_stop_rule_reads_all_three_gates_and_the_36_of_40_rung'),
]
def run(td, test):
    r = subprocess.run([PY, '-I', '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'part218/' + test],
                       cwd=td, capture_output=True, text=True)
    return r.returncode, (r.stdout.strip().splitlines() or ['?'])[-1]

for fname, old, new, test in cases:
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / 'part218'
        shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns('__pycache__', 'results', '.pytest_cache'))
        rc0, last0 = run(td, test)          # control: the unbroken copy must pass, and must have run one test
        p = dst / fname; s = p.read_text()
        assert s.count(old) == 1, (fname, old[:40], s.count(old))
        p.write_text(s.replace(old, new))
        rc, last = run(td, test)
        ok_control = rc0 == 0 and ' passed' in last0
        red = rc == 1 and ' failed' in last
        print(('RED  ' if (red and ok_control) else 'CHECK') + f' {fname}: {test.split("::")[1]} | control: {last0} | broken: {last}', flush=True)
