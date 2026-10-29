"""Checks for Part 107, written before any run: the bf16 step counter and the scorer's verdicts on fixtures."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score107 as s  # noqa: E402
from where107 import bf16_steps  # noqa: E402


def test_bf16_steps_counts_adjacent_values():
    a = torch.tensor([1.0, -1.0, 0.5, 2.0], dtype=torch.bfloat16)
    up = torch.nextafter(a.float(), torch.full_like(a.float(), 10.0)).to(torch.bfloat16)  # rounds back to a
    assert torch.equal(bf16_steps(a, up), torch.zeros(4, dtype=torch.int32))
    ai = a.view(torch.int16)
    b = (ai + 1).view(torch.bfloat16)  # one representable step away from zero for positives, toward for negatives
    assert torch.equal(bf16_steps(a, b), torch.ones(4, dtype=torch.int32))
    assert bf16_steps(torch.tensor([1.0], dtype=torch.bfloat16), torch.tensor([-1.0], dtype=torch.bfloat16)).item() == 2


def fixture(dpo_rel, one_step, slot_r, concept_r, random_r):
    units = [dict(id=f"s{i}", kind="slot", layer=3) for i in range(21)] + \
            [dict(id=f"c{i}", kind="concept", layer=3) for i in range(18)]
    def stage(rel, share, sr, cr):
        return {"groups": {"attn": {"rel_change": rel}, "mlp": {"rel_change": rel}},
                "precision_floor": {"changed_frac": 0.1, "one_step_frac_of_changed": share},
                "unit_R": {**{f"s{i}": sr for i in range(21)}, **{f"c{i}": cr for i in range(18)}},
                "random_R_by_layer": {"3": [random_r] * 64}}
    return {"units": units, "stages": {"sft_change": stage(0.1, 0.2, 1.0, 1.0),
                                       "dpo_change": stage(dpo_rel, one_step, slot_r, concept_r),
                                       "rl_change": stage(dpo_rel, one_step, 1.0, 1.0)}}


def test_targeted_dpo_change_reads_as_aimed():
    out = s.score(fixture(dpo_rel=0.001, one_step=0.2, slot_r=2.0, concept_r=1.0, random_r=1.0))
    assert out["107.1"]["verdict"] == "held" and out["107.2"]["verdict"] == "failed"
    assert out["107.3"]["verdict"] == "held" and out["reading"].startswith("DPO's change is aimed")


def test_rounding_floor_blocks_the_reading():
    out = s.score(fixture(dpo_rel=0.001, one_step=0.9, slot_r=2.0, concept_r=1.0, random_r=1.0))
    assert out["107.2"]["verdict"] == "held" and out["reading"].startswith("107.3 reads rounding")


def test_diffuse_change_fails_107_3():
    out = s.score(fixture(dpo_rel=0.05, one_step=0.2, slot_r=1.0, concept_r=1.0, random_r=1.0))
    assert out["107.1"]["verdict"] == "failed" and out["107.3"]["verdict"] == "failed"
