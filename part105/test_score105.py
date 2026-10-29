"""Fixture checks for part105/score105.py (run before Part 105's labels existed, now kept as a test): with SFT's own
labels standing in for the extrapolated run the scorer must read F; with inst_none's it must read D."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def fixture(tmp, ext_from):
    for rel in ("results/part92/step2/kinds.json", "results/part92/step2/neurons.json",
                "results/part99/labels-inst_none.jsonl", "results/part103/labels-attnSFT_all.jsonl"):
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp / rel)
    (tmp / "results/part105").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / ext_from, tmp / "results/part105/labels-attnEXT_all.jsonl")
    subprocess.run([sys.executable, "-I", str(REPO / "part105/score105.py"), str(tmp)], check=True, capture_output=True)
    return json.loads((tmp / "results/part105/score105.json").read_text())["predictions"]


def test_sft_labels_as_ext_read_as_fragility(tmp_path):
    p = fixture(tmp_path, "results/part103/labels-attnSFT_all.jsonl")
    assert p["105.1"]["verdict"] == "failed" and p["105.1"]["reading"].startswith("F")


def test_inst_none_labels_as_ext_read_as_dependence(tmp_path):
    p = fixture(tmp_path, "results/part99/labels-inst_none.jsonl")
    assert p["105.1"]["verdict"] == "held" and p["105.1"]["reading"].startswith("D")
