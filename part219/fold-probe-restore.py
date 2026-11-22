"""After a transplant run and after an ablation run in one process: are every host parameter's bytes
back, and are the hooks gone (so a later run cannot inherit them)?
"""
import argparse
import json
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
OUT = Path(tempfile.mkdtemp(prefix="fold-restore-"))
OUT.mkdir(parents=True, exist_ok=True)
replay = OUT / "replay.json"
if replay.exists():
    replay.unlink()


def make(tag, **overrides):
    values = dict(host=str(ROOT / "host"), donor=str(ROOT / "donor"), swap="none", tokenizer=None,
                  neurons=str(ROOT / "neurons.json"), frame="chat", tag=tag, out=str(OUT),
                  device="cpu", batch=8, kinds=None, only=None, pushes=str(ROOT / "pushes.pt"),
                  limit=None, layers=None, extrapolate=False, questions=None,
                  attn_implementation="eager", ablate_heads=None, transplant_heads=None,
                  transplant_mode=None, reference_replay=str(replay), record_token_ids=True)
    values.update(overrides)
    return argparse.Namespace(**values)


def snapshot(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def differing(before, model):
    return [name for name, parameter in model.named_parameters()
            if not torch.equal(parameter, before[name])]


def hooks_on(model):
    return sum(len(layer.self_attn.o_proj._forward_pre_hooks) + len(layer.self_attn.o_proj._forward_hooks)
               for layer in model.model.layers)


tokenizer, model, donor = None, None, None
host_tok, model = c.load_model(str(ROOT / "host"), "cpu", attn_implementation="eager")
tokenizer = host_tok
donor = swap99.load_donor(str(ROOT / "donor"), attn_implementation="eager")
base = snapshot(model)

print("hooks before any run:", hooks_on(model))
swap99.run(make("T1", transplant_heads="L0H1,L2H0", transplant_mode="donor"), model=model,
           tokenizer=tokenizer, donor_model=donor)
print("after transplant: changed parameters =", differing(base, model), "hooks =", hooks_on(model))
meta = json.loads((OUT / "T1.meta.json").read_text())
print("T1 transplant records:", [(r["parameter"], r["slice"], r["matches_expected"])
                                 for r in meta["intervention_receipt"]["transplant"]["records"]][:3], "...",
      len(meta["intervention_receipt"]["transplant"]["records"]), "records")
print("T1 reset:", meta["reset_receipt"]["matches_reference"])

swap99.run(make("A1", ablate_heads="L0H1,L1H0,L2H0"), model=model, tokenizer=tokenizer)
print("after ablation: changed parameters =", differing(base, model), "hooks =", hooks_on(model))
meta = json.loads((OUT / "A1.meta.json").read_text())
print("A1 ablation records:", [(r["layer"], r["head"], r["slice"], r["all_zero"])
                               for r in meta["intervention_receipt"]["ablation"]])
print("A1 reset:", meta["reset_receipt"]["matches_reference"])

swap99.run(make("T2", transplant_heads="L0H1,L2H0", transplant_mode="opposite"), model=model,
           tokenizer=tokenizer, donor_model=donor)
print("after opposite transplant: changed parameters =", differing(base, model), "hooks =", hooks_on(model))
meta = json.loads((OUT / "T2.meta.json").read_text())
print("T2 reset:", meta["reset_receipt"]["matches_reference"])

again = swap99.replay_receipt(model, tokenizer, "chat", "cpu")
print("pristine replay equals stored reference:", swap99.compare_replay(again, json.loads(replay.read_text())) is None)
