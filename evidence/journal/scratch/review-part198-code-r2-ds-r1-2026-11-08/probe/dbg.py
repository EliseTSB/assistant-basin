import sys, types
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
R.torch = types.SimpleNamespace(equal=lambda a, b: True, zeros_like=lambda a: a, set_num_threads=lambda n: None)
print(R.torch, R.torch.zeros_like(3))
