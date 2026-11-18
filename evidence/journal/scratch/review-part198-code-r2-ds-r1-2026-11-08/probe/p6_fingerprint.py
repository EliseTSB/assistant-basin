"""Q1: tensor_fingerprint / model_state_snapshot on the REAL run198.
(a) view / non-contiguous / transposed tensors, (b) same-storage mutation after snapshot,
(c) a parameter replaced by a new tensor with the same bytes, (d) a dtype change."""
import sys
import torch
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R

t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("(a) contiguous        ", R.tensor_fingerprint(t)[:12])
print("(a) .T (non-contig)   ", R.tensor_fingerprint(t.T)[:12], "== contiguous copy?", R.tensor_fingerprint(t.T) == R.tensor_fingerprint(t.T.contiguous()))
print("(a) view of t[0:2]    ", R.tensor_fingerprint(t[0:2])[:12], "== copy?", R.tensor_fingerprint(t[0:2]) == R.tensor_fingerprint(t[0:2].clone()))
print("(a) shape in digest   ", R.tensor_fingerprint(t) != R.tensor_fingerprint(t.reshape(4, 3)))
b = torch.tensor([1.0, 2.0, 3.0])
print("(a) dtype in digest   ", R.tensor_fingerprint(b) != R.tensor_fingerprint(b.double()))
print("(a) 0-d tensor raises:", end=" ")
try:
    R.tensor_fingerprint(torch.tensor(1.0)); print("no")
except RuntimeError as exc:
    print(type(exc).__name__, exc)
print("(a) bfloat16          ", R.tensor_fingerprint(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))[:12])

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.register_buffer("b", torch.tensor([[1.0, 2.0]]))

m = M()
snap = R.model_state_snapshot(m)
print("(b) keys              ", sorted(snap))
with torch.no_grad():
    m.w[1] = 9.0                      # in-place through the same storage
print("(b) in-place edit detected:", not R.model_matches_snapshot(m, snap))
m.w.data = torch.tensor([1.0, 2.0, 3.0])   # restore
print("(b) restored matches  ", R.model_matches_snapshot(m, snap))
m.w = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))  # new object, same bytes
print("(c) replaced param, same bytes still matches:", R.model_matches_snapshot(m, snap))
with torch.no_grad():
    m.b.mul_(1.0)                     # no-op write
print("(b) no-op write matches:", R.model_matches_snapshot(m, snap))
m.b = m.b.half()
print("(d) dtype change detected:", not R.model_matches_snapshot(m, snap))
# A parameter that is aliased twice (tied weights) is hashed twice under two names -- fine.
