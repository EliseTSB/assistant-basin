import json, os, sys, subprocess
sys.path.insert(0, os.getcwd())
os.environ.setdefault("QWEN35_SNAPSHOT", "/home/elisepc/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
import torch
SNAP=os.environ["QWEN35_SNAPSHOT"]
def tiny_model(dirname):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(SNAP, "config.json"), encoding="utf-8"))
    t = dict(cfg.get("text_config", cfg))
    t.update(num_hidden_layers=4, hidden_size=64, intermediate_size=96, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
             linear_key_head_dim=16, linear_value_head_dim=16, layer_types=t["layer_types"][:4],
             tie_word_embeddings=True)
    t.pop("architectures", None)
    conf = AutoConfig.for_model(t.get("model_type", "qwen3_5_text"), **{k: v for k, v in t.items() if k != "model_type"})
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32)
    model.save_pretrained(dirname)
    AutoTokenizer.from_pretrained(SNAP).save_pretrained(dirname)
S = sys.argv[1]
mdir = os.path.join(S, "tiny"); tiny_model(mdir)
corpus = os.path.join(S, "pile.jsonl")
with open(corpus, "w", encoding="utf-8") as f:
    for i in range(400):
        f.write(json.dumps({"text": f"Document {i} talks about gardens, roads and weather in some town. " * 6}) + "\n")
base = [sys.executable, "part20c.py", "--model", mdir, "--device", "cpu", "--layer", "2", "--neuron", "5", "--corpus", corpus]
r = subprocess.run(base + ["--out", os.path.join(S, "o1")], capture_output=True, text=True)
print("run1 rc", r.returncode, r.stdout[-200:], r.stderr[-300:])
assert r.returncode != 0 and "self-check failed" in (r.stdout + r.stderr), "self-check did not refuse"
import re
fm, sm = re.findall(r"filler max ([-0-9.e]+), summary max ([-0-9.e]+)", r.stdout + r.stderr)[0]
r2 = subprocess.run(base + ["--out", os.path.join(S, "o2"), "--expect-filler-max", fm, "--expect-summary-max", sm], capture_output=True, text=True)
print("run2 rc", r2.returncode, r2.stderr[-300:])
d = json.load(open(os.path.join(S, "o2", "pertoken.json")))
assert set(d["inputs"]) == {"filler", "filler_summary", "filler_neutral", "lead_summary", "lead_neutral"}
fs = d["inputs"]["filler_summary"]
assert "".join(fs["tokens"][i] for i in fs["target_positions"]).strip().startswith("Several"), "summary span wrong"
assert len(d["inputs"]["filler"]["values"]) == d["filler_token_count"]
print("SMOKE PASS", json.dumps(d["summary"])[:300])
