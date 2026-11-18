import json
import os
import sys
import tempfile
import types
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-48b")
import part48b

class FakeModel:
    config = types.SimpleNamespace(_commit_hash="fake-rev")
    def eval(self):
        return self
class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return FakeModel()
class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return object()
transformers = types.ModuleType("transformers")
transformers.AutoModelForCausalLM = FakeAutoModel
transformers.AutoTokenizer = FakeAutoTokenizer
modeling_utils = types.ModuleType("transformers.modeling_utils")
transformers.modeling_utils = modeling_utils
sys.modules["transformers"] = transformers
sys.modules["transformers.modeling_utils"] = modeling_utils
part48b.p48.MODELS["fake"] = "org/fake"
part48b.score_model = lambda *args, **kwargs: []
part48b.summarize = lambda *args, **kwargs: {}

def fail_after_prefix(obj, fp, indent=None):
    fp.write('{"partial":')
    fp.flush()
    raise OSError("simulated disk-full")
part48b.json.dump = fail_after_prefix

with tempfile.TemporaryDirectory(dir="/home/elisepc/green/Journal/scratch/review-part48b-r1-luna-luna-r1-2026-10-21") as out:
    sys.argv = ["part48b", "--models", "fake", "--out", out]
    try:
        part48b.main()
    except OSError as exc:
        print("raised:", repr(exc))
    path = os.path.join(out, "fake.json")
    payload = open(path, encoding="utf-8").read()
    print("file remains:", os.path.exists(path), "bytes:", len(payload), "content:", repr(payload))
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        print("valid JSON:", False, "decode error:", exc.msg)
