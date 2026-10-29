"""Fixture check for score106b.py, written before any Part 106b run."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import score106b as s  # noqa: E402


def test_competitor_drop_reads_as_lowering(tmp_path):
    ids = [f"u{i}" for i in range(13)]
    personas = {i: (s.SIX[k] if k < 6 else f"p{k}") for k, i in enumerate(ids)}
    for c, comp_push, comp_none, pers in (("sft", -10.0, -20.0, -12.0), ("dpo", -14.0, -23.0, -12.0)):
        (tmp_path / f"lp106b-{c}.jsonl").write_text("".join(
            json.dumps({"id": i, "unit": u, "lp_sum": v}) + "\n" for i in ids for u, v in (("neuron", comp_push), ("none", comp_none))))
        (tmp_path / f"lp-{c}.jsonl").write_text("".join(json.dumps({"id": i, "unit": "neuron", "lp_sum": pers}) + "\n" for i in ids))
    out = s.score(tmp_path, {i: False for i in ids}, personas)
    assert out["106b.1"]["verdict"] == "held" and out["106b.2"]["verdict"] == "held" and out["106b.3"]["verdict"] == "held"
    assert out["reading"].startswith("DPO switched") and "in general" in out["reading"]
