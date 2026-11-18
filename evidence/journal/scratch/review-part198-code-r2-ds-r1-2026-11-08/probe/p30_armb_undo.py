"""Arm B: the case row is appended AFTER w.undo() and carries undo_zero_exact, so a crash between
the write and the undo leaves only a failure row (retried next run), not a case row. Confirm by
reading the order and by driving run_arm_b with a stub WSP whose undo() raises."""
import json, sys, tempfile
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R

src = open(HEAD + "/run198.py").read()
seg = src[src.index("def run_arm_b("):src.index("def run_arm_b_seq(")]
print("undo() before the case append:", seg.index("undone = w.undo()") < seg.index("append_jsonl(log, record)"))
print("undo_zero_exact filled before the append:", seg.index('record["undo_zero_exact"]') < seg.index("append_jsonl(log, record)"))
