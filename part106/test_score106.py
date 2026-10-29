"""Fixture checks for part106/score106.py, written before any Part 106 run: a synthetic sharpening pattern must read S,
a creation pattern C, and random pushes that move the sentence must be reported without blocking the reading."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score106 as s  # noqa: E402


def write(tmp, sft_gap, dpo_gap, random_shift=0.0, n_slot=5, n_concept=4):
    for ckpt in s.CKPTS:
        rows = []
        for k, n in (("slot", n_slot), ("concept", n_concept)):
            for j in range(n):
                i = f"{k}-{j}"
                g = {"sft": sft_gap, "dpo": dpo_gap, "instruct": dpo_gap}[ckpt] if k == "slot" else 0.5
                base = -30.0
                for alpha, scale in ((0.4, 0.5), (0.8, 1.0), (1.6, 2.0)):
                    rows.append(dict(id=i, kind=k, layer=3, neuron=j, target="t", unit="random", alpha=alpha,
                                     lp_sum=base + random_shift))
                    rows.append(dict(id=i, kind=k, layer=3, neuron=j, target="t", unit="neuron", alpha=alpha,
                                     lp_sum=base + random_shift + g * scale))
                rows.append(dict(id=i, kind=k, layer=3, neuron=j, target="t", unit="none", alpha=0.0, lp_sum=base))
        (tmp / f"lp-{ckpt}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_sharpening_reads_s(tmp_path):
    write(tmp_path, sft_gap=3.0, dpo_gap=6.0)
    out = s.score(tmp_path)
    assert out["106.1"]["verdict"] == "held" and out["reading"].startswith("S")
    assert out["106.2"]["verdict"] == "held"  # slot growth 3 vs concept 0
    # gap(sft, 1.6) = 6 equals gap(dpo, 0.8) = 6 for all 5 slot units, but the rule needs 11 units
    assert out["106.4"]["slot_units_sft_1.6_at_least_dpo_0.8"] == 5 and out["106.4"]["verdict"] == "failed"


def test_creation_reads_c(tmp_path):
    write(tmp_path, sft_gap=0.1, dpo_gap=5.0)
    out = s.score(tmp_path)
    assert out["106.1"]["verdict"] == "failed" and out["reading"].startswith("C")


def test_random_shift_is_reported_not_blocking(tmp_path):
    write(tmp_path, sft_gap=3.0, dpo_gap=6.0, random_shift=2.5)
    out = s.score(tmp_path)
    assert out["106.5"]["verdict"] == "failed"
    assert out["reading"].startswith("S") and "loosens" in out["reading"]
