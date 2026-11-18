"""Probe: (a) does a write to a parameter the editor did NOT report go unnoticed by the restoration check?
   (b) is the 'base' copy an alias of the live weights?"""
import sys
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import torch
import run198 as R

# (a) the restoration check in run198.py:831 iterates only over `orig` (the editor's returned dict).
model = torch.nn.Linear(4, 4)
model.weight.data = model.weight.data.clone()
params = dict(model.named_parameters())
W0 = params["weight"].detach().clone()
orig = {"weight": W0}                      # editor reports only 'weight'
with torch.no_grad():
    params["weight"].copy_(W0)             # restore what it reported
with torch.no_grad():
    params["bias"].add_(1.0)               # a write the editor did not report
restored = all(torch.equal(params[p].detach(), w0.to(params[p].dtype)) for p, w0 in orig.items())
print("(a) restoration check says restored =", restored, "; bias actually changed:",
      not torch.equal(params["bias"].detach(), torch.zeros(4)))

# (b) gpt2_geometry takes W = params[pname].detach().T.contiguous(); is that a copy?
t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
W = t.detach().T.contiguous()
print("(b) W shares storage with the parameter:", W.data_ptr() == t.data_ptr())
# and load_gpt2_model replaces every p.data with a clone, so base stats are computed on the clone.
