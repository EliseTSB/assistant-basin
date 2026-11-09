# Throwaway end-to-end harness for Part 200 (review r1). Real: run200.run, run163.run, pushes_for, chunker,
# batch-1 stage, record writer, Gates R1/R3, common86 vector helpers. Fake: the model (neuron_report.load_model and
# find_text_layers) and the generator (the Part 160 call), which returns text that encodes the push each row received.
import sys, json, hashlib, types, tempfile, traceback
from pathlib import Path
from types import SimpleNamespace
REPO = Path("/home/elisepc/green/worktrees/ab-part200")
for p in (str(REPO), str(REPO / "part86")):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
import neuron_report as nr
from part200 import run200
from part163 import run163
from part155 import run155

WIDTH, HIDDEN = 9216, 16
g = torch.Generator().manual_seed(7)
W = torch.randn(HIDDEN, WIDTH, generator=g)
LAYERS = [SimpleNamespace(mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=W))) for _ in range(40)]

class FakeModel:
    config = SimpleNamespace(_attn_implementation="eager")
    def requires_grad_(self, flag): return self
    def eval(self): return self

def fake_load_model(model_id, revision, device, attn_implementation="eager", dtype=None):
    nr.LAST_LOAD = {"attn_implementation": "eager", "attn_implementation_requested": "eager", "dtype": "torch.bfloat16",
                    "device": str(device), "torch": "x", "transformers": "y"}
    return object(), FakeModel()

nr.load_model = fake_load_model
nr.find_text_layers = lambda model: (None, LAYERS, None)

def push_hash(pushes, r):
    h = hashlib.sha256()
    for layer in sorted(pushes):
        vec = pushes[layer][r].detach().cpu().contiguous()
        if float(vec.abs().sum()) > 0:          # only the row's own layer; zero rows for other layers are not its push
            h.update(f"L{layer}:".encode()); h.update(vec.numpy().tobytes())
    return h.hexdigest()[:12]

class Gen:
    """Fake Part 160 call. kill_at = call number (1-based, over all generate calls in this launch) that raises."""
    def __init__(self, kill_at=None, kill_when=None):
        self.n = 0; self.kill_at = kill_at; self.kill_when = kill_when; self.log = []
    def __call__(self, model, tok, layers, question, pushes, max_new_tokens, device):
        self.n += 1
        if self.kill_at is not None and self.n == self.kill_at:
            raise RuntimeError("simulated kill (SIGKILL-like)")
        if not pushes:
            hashes = ["plain"]
        else:
            n = next(iter(pushes.values())).shape[0]
            hashes = [push_hash(pushes, r) for r in range(n)]
        self.log.append({"q": question[:12], "n": len(hashes)})
        return [{"answer": f"Answer {h} alpha beta gamma {question[:6]} delta", "token_count": 80} for h in hashes]

UNITS = str(REPO / "part163/units163.json")

def args_for(folder):
    return SimpleNamespace(device="cpu", model="stub", units=UNITS, out=str(Path(folder) / "answers.jsonl"), batch_size=64)

def launch(folder, gen, loader=True):
    a = args_for(folder)
    loader_obj = run200.eager_loader(a) if loader else None
    return run200.run(a, loader=loader_obj, generate_fn=gen)

def expected_hashes():
    """The push each unit should receive, built alone (one row), by the same run163 code path."""
    units = json.loads(Path(UNITS).read_text())
    common = run155._common()
    medians = json.loads((REPO / "results/part112/medians.json").read_text())
    cache = {}
    out = {}
    for u in units:
        row = {**run163._unit_row(u), "question_index": 0}
        pushes = run163.pushes_for([row], LAYERS, medians, common, cache)
        out[u["unit_id"]] = push_hash(pushes, 0)
    return out, units

def read_rows(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
