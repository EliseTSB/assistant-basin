"""Failure paths: a raise between the model mutation and the generation try/finally.

Case A: the post-transplant attention_probe raises -> is the donor's transplant left in the host?
Case B: residual_medians raises after --swap attn -> are the swapped parameters left in the host?
"""
import argparse
import sys
from pathlib import Path

import torch

WORKTREE = str(Path(__file__).resolve().parents[1])  # the worktree this copy lives in
sys.path.insert(0, WORKTREE)
sys.path.insert(0, WORKTREE + "/part92")
sys.path.insert(0, WORKTREE + "/part99")

import common92 as c  # noqa: E402
import swap99  # noqa: E402

ROOT = Path("/home/elisepc/green/Journal/scratch/review-part219-code-r1-ds-r1-2026-11-22/exp/world")
import tempfile
OUT = Path(tempfile.mkdtemp(prefix="fold-fail-"))
OUT.mkdir(parents=True, exist_ok=True)


def make(tag, **overrides):
    values = dict(host=str(ROOT / "host"), donor=str(ROOT / "donor"), swap="none", tokenizer=None,
                  neurons=str(ROOT / "neurons.json"), frame="chat", tag=tag, out=str(OUT),
                  device="cpu", batch=8, kinds=None, only=None, pushes=str(ROOT / "pushes.pt"),
                  limit=None, layers=None, extrapolate=False, questions=None,
                  attn_implementation="eager", ablate_heads=None, transplant_heads=None,
                  transplant_mode=None, reference_replay=None, record_token_ids=True)
    values.update(overrides)
    return argparse.Namespace(**values)


def differing(before, model):
    return [name for name, parameter in model.named_parameters() if not torch.equal(parameter, before[name])]


tokenizer, model = c.load_model(str(ROOT / "host"), "cpu", attn_implementation="eager")
donor = swap99.load_donor(str(ROOT / "donor"), attn_implementation="eager")
base = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

# Case A: second probe raises --------------------------------------------------------------------
real_probe = swap99.attention_probe
calls = {"n": 0}


def flaky_probe(*args, **kwargs):
    calls["n"] += 1
    if calls["n"] == 2:
        raise RuntimeError("forced probe failure")
    return real_probe(*args, **kwargs)


swap99.attention_probe = flaky_probe
try:
    swap99.run(make("FA", transplant_heads="L0H1,L2H0", transplant_mode="donor"),
               model=model, tokenizer=tokenizer, donor_model=donor)
except RuntimeError as exc:
    print("case A raised:", exc)
finally:
    swap99.attention_probe = real_probe
changed = differing(base, model)
print("case A: parameters left changed after the raise:", changed)
for name, parameter in model.named_parameters():
    parameter.copy_(base[name])
print("case A: model reset by the harness for the next case:", not differing(base, model))

# Case B: residual_medians raises after --swap attn ----------------------------------------------
real_medians = c.residual_medians


def flaky_medians(*args, **kwargs):
    raise RuntimeError("forced medians failure")


c.residual_medians = flaky_medians
try:
    swap99.run(make("FB", swap="attn", layers="0-1"), model=model, tokenizer=tokenizer, donor_model=donor)
except RuntimeError as exc:
    print("case B raised:", exc)
finally:
    c.residual_medians = real_medians
changed = differing(base, model)
print("case B: parameters left changed after the raise:", changed[:6], "...", len(changed), "total")
