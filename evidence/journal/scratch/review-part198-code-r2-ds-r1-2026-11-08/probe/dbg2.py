import sys, types
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
R.torch = types.SimpleNamespace(equal=lambda a, b: True, zeros_like=lambda a: a, set_num_threads=lambda n: None)
src = open("probe/p32_armb_retry.py").read()
src = src.replace("import run198 as R\n", "import run198 as R\n", 1)
# execute p32 but stop just before run_arm_b
head = src.split("out = Path(tempfile.mkdtemp")[0]
exec(head)
print("after p32 head, R.torch is", R.torch)
