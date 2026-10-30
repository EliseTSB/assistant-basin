"""One-off profile of Part 125's call path on the A770 (diagnostic, not part of the registered run)."""
import os, sys
if "--copy-engine" in sys.argv:
    os.environ["UR_L0_USE_COPY_ENGINE"] = "1"
    os.environ["SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE"] = "1"
import cProfile, io, json, pstats, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for d in (HERE, ROOT, ROOT / "part92", ROOT / "part119", ROOT / "part123", ROOT / "part92b"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
import torch
import common92
import session125 as session
from transformers import DynamicCache

def sync(x):
    return float(x.flatten()[0].float().item())

def main():
    snap = sys.argv[sys.argv.index("--model") + 1]
    device = sys.argv[sys.argv.index("--device") + 1]
    n = int(sys.argv[sys.argv.index("--calls") + 1]) if "--calls" in sys.argv else 240
    t0 = time.perf_counter()
    tok, model = common92.load_model(snap, device)
    print("load", round(time.perf_counter() - t0, 2), flush=True)
    print("attn", getattr(model.config, "_attn_implementation", None), flush=True)
    ids = tok.apply_chat_template([{"role": "user", "content": "Who are you?"}], add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)
    def run(label, calls, hash_each=True, use_mask=True):
        cache = DynamicCache()
        cur = list(ids)
        with torch.inference_mode():
            inp = torch.tensor([cur], device=device)
            out = model(input_ids=inp, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            sync(out.logits)
            times = {"forward": 0.0, "sync": 0.0, "hash": 0.0}
            token = int(out.logits[0, -1].float().argmax().item())
            t00 = time.perf_counter()
            for i in range(calls):
                length = cache.get_seq_length()
                inp = torch.tensor([[token]], device=device)
                kw = {}
                if use_mask:
                    kw = dict(attention_mask=torch.ones((1, length + 1), device=device, dtype=torch.long),
                              position_ids=torch.tensor([[length]], device=device))
                a = time.perf_counter()
                out = model(input_ids=inp, past_key_values=cache, use_cache=True, **kw)
                b = time.perf_counter()
                logits = out.logits[:, -1, :]
                token = int(logits[0].float().argmax().item())
                c = time.perf_counter()
                if hash_each:
                    session.logit_hash(logits)
                d = time.perf_counter()
                cache = out.past_key_values
                times["forward"] += b - a
                times["sync"] += c - b
                times["hash"] += d - c
            total = time.perf_counter() - t00
        print(json.dumps({"label": label, "calls": calls, "calls_per_s": round(calls / total, 2),
                          "ms": {k: round(1000 * v / calls, 1) for k, v in times.items()}}), flush=True)
    run("warm", 40)
    run("full", n)
    run("no hash", n, hash_each=False)
    run("no mask", n, hash_each=False, use_mask=False)
    pr = cProfile.Profile()
    pr.enable()
    run("profiled", 80, hash_each=True)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
    print("\n".join(line[:110] for line in s.getvalue().splitlines()[4:36]), flush=True)
    print("PROFILE125 DONE", flush=True)

main()
