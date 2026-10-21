"""Part 43 batched-vs-single diagnostic (2026-10-21). Does not modify part43.py: imports its own _sequences,
_pad_records and captured_forward, and loads Qwen3.5-4B (pinned revision) once per dtype the same way
part39.load_model_direct does, but with the dtype chosen here (part43 itself accepts only bf16).
Reports, per record, the check's own statistic (max abs logit diff over [:length] x vocab), where that max sits, and at
the score position: max abs diff over the vocab, the top-5 tokens' logit diffs, and whether the argmax agrees."""
import json, os, sys, time
import torch
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "/root/hf-cache"
sys.path.insert(0, os.getcwd())
import part43
MODEL, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils as _mu
if hasattr(_mu, "caching_allocator_warmup"): _mu.caching_allocator_warmup = lambda *a, **k: None
out = {"model": MODEL, "revision": REV, "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
seqs = part43._sequences(tok); records = [seqs[0], seqs[2]]
for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    kw = dict(revision=REV, dtype=dtype, low_cpu_mem_usage=True, device_map={"": "cuda"})
    try: model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText; model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
    model.eval(); torch.set_grad_enabled(False)
    bo, _ = part43.captured_forward(model, tok, records)
    bl = (bo.logits if hasattr(bo, "logits") else bo[0]).float()
    rows = []
    for i, rec in enumerate(records):
        so, _ = part43.captured_forward(model, tok, [rec])
        sl = (so.logits if hasattr(so, "logits") else so[0]).float()[0]
        L = len(rec["input_ids"]); d = (bl[i, :L] - sl[:L]).abs()
        flat = int(d.argmax()); pos, vid = divmod(flat, d.shape[1])
        sp = int(rec.get("score_position", L - 1))
        top = torch.topk(sl[sp], 5).indices.tolist()
        rows.append({"record": i, "length": L, "score_position": sp,
                     "check_stat_max_abs_diff": float(d.max()), "max_at_position": pos, "max_at_token": tok.decode([vid]),
                     "max_abs_diff_per_position_median": float(d.max(dim=1).values.median()),
                     "score_position_max_abs_diff": float(d[sp].max()),
                     "score_position_top5": [{"token": tok.decode([t]), "single": float(sl[sp, t]), "batched": float(bl[i, sp, t]),
                                              "abs_diff": float(abs(sl[sp, t] - bl[i, sp, t]))} for t in top],
                     "argmax_agrees": int(sl[sp].argmax()) == int(bl[i, sp].argmax())})
    tol = 0.1 if name == "bf16" else 1e-4
    out[name] = {"rows": rows, "check_tolerance_in_part43": tol,
                 "would_pass_check": all(r["check_stat_max_abs_diff"] <= tol for r in rows)}
    print(name, json.dumps(out[name])[:600], flush=True)
    del model; torch.cuda.empty_cache()
json.dump(out, open("diag-bf16-score-position.json", "w"), indent=2); print("DIAG DONE")
