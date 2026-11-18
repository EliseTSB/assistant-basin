"""Arm B resume after a crash inside a case: drive the REAL run_arm_b with a stub WSP whose
undo() raises on the first run. Does the case get retried (failure row only), or is a case row
left behind that would suppress the retry?"""
import json, sys, tempfile, types
from pathlib import Path
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R
R.torch = types.SimpleNamespace(equal=lambda a, b: True, zeros_like=lambda a: a, set_num_threads=lambda n: None)

class T:
    n = 4
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def double(self): return self
    def clone(self): return self
    def zero_(self): return self
    def zeros_like(self): return self
    def to(self, *a, **k): return self
    def __add__(self, o): return self
    def __sub__(self, o): return self
    def __mul__(self, o): return self
    def __rmul__(self, o): return self
    def __truediv__(self, o): return self
    def T(self): return self
    def norm(self): return 1.0
    def mean(self): return 0.0
    def __getitem__(self, i): return self
    def __len__(self): return self.n
    def __eq__(self, o): return False

class Tok:
    def decode(self, ids): return "word"

class StubWSP:
    Ls = [0, 1]
    layers = {0: types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(weight=T()))),
              1: types.SimpleNamespace(mlp=types.SimpleNamespace(down_proj=types.SimpleNamespace(weight=T())))}
    g, head, tok = T(), T(), Tok()
    delta = {0: T(), 1: T()}
    facts = []
    fail_undo = True
    def enc(self, s): return [1]
    def answer(self, q, edited, n=24): return "x"
    def teach(self, asks, obj):
        self.facts.append(1)
        return {"free_exact": "3/3"}
    def undo(self):
        if self.fail_undo:
            raise RuntimeError("simulated undo failure")
        self.facts.clear()
        return []

stub = StubWSP()
R.ensure_offline = lambda: None
R.torch = types.SimpleNamespace(equal=lambda a, b: True, zeros_like=lambda a: a, set_num_threads=lambda n: None)
R.root_out_dir = lambda p: Path(p).expanduser().resolve()
R.load_wsp_module = lambda: types.SimpleNamespace(WSP=lambda path, device: stub)
R.target_info_wsp = lambda w, text, S: {"target_ids": [1], "eligible_target_ids": [1], "name_eligible": True}
R.build_wsp_base = lambda w, S: {l: {"f1": 1.0, "f2": 1.0, "pcs": 1.0, "t_max": 1.0, "t_word": 1.0, "pairs": []} for l in (0, 1)}
R.stats_for = lambda W, g, c, E, dec, S, pcs=True, k=64: {"f1": 1.0, "f2": 1.0, "pcs": 1.0, "t_max": 1.0, "t_word": 1.0, "pairs": []}
R.pair_by_layer = lambda base, current, S: {l: [] for l in base}
R.update_metrics = lambda delta, base: {"delta_spectral": 1.0}
R.pair_to_dict = lambda p, decode=None: {"fake": True}
R.random_control_layers = lambda base, deltas, seed: {l: deltas[l] for l in deltas}

out = Path(tempfile.mkdtemp(prefix="armbretry-")) / "out"
out.mkdir()
R.run_arm_b(1, out)                      # smoke, one case: undo fails
stub.fail_undo = False
import torch as _real_torch
R.torch = _real_torch
import torch as _real_torch
R.torch = types.SimpleNamespace(equal=_real_torch.equal, zeros_like=_real_torch.zeros_like, set_num_threads=lambda n: None)
R.run_arm_b(1, out)                      # smoke retry, undo now works
rows = [json.loads(l) for l in (out / "armB-wsp.jsonl").read_text().splitlines()]
for r in rows:
    print(r["record_type"], r.get("run_mode"), r.get("case_id"), r.get("phase"), r.get("undo_zero_exact"))
