#!/usr/bin/env python3
"""Part 151: top activating windows for Part 150's five units, plus the alien neuron, on Qwen3.5-4B itself."""
import argparse
import heapq
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "part86")):
    if p not in sys.path:
        sys.path.insert(0, p)

UNITS = ((19, 4979), (19, 5975), (22, 2742), (23, 116), (24, 6835), (24, 8542))
MODEL_ID, REVISION = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_TOKENS, TOP_N, BEFORE, AFTER = 256, 20, 30, 12


def unit_name(layer, neuron):
    return f"L{layer}N{neuron}"


def offer(heap, value, item, keep=TOP_N):
    """Keep the largest values; item is (doc, position)."""
    if len(heap) < keep:
        heapq.heappush(heap, (value, item))
    elif value > heap[0][0]:
        heapq.heapreplace(heap, (value, item))


def window(ids, pos, decode):
    """Text around one token, the token marked, with the words after it kept apart for the scorer."""
    before = decode(ids[max(0, pos - BEFORE):pos])
    mark = decode(ids[pos:pos + 1])
    after = decode(ids[pos + 1:pos + 1 + AFTER])
    return {"before": before, "mark": mark, "after": after}


def collect(model, tok, downs, docs, batch_size, device, torch, log=print):
    """One pass over docs; per batch only the unit columns cross to the CPU."""
    seen = {}
    handles = []
    for layer in sorted({l for l, _ in UNITS}):
        neurons = [n for l, n in UNITS if l == layer]

        def grab(_m, args, layer=layer, neurons=neurons):
            seen[layer] = (args[0][:, :, neurons].float().cpu().numpy(), neurons)
        handles.append(downs[layer].register_forward_pre_hook(grab))
    tops = {unit_name(l, n): [] for l, n in UNITS}
    lows = {unit_name(l, n): [] for l, n in UNITS}
    all_ids = {}
    tokens = 0
    t0 = time.time()
    try:
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            enc = tok([d["text"] for d in batch], return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
            ids = enc["input_ids"]
            mask = enc["attention_mask"].numpy().astype(bool)
            with torch.inference_mode():
                model(input_ids=ids.to(device), attention_mask=enc["attention_mask"].to(device), use_cache=False)
            for row, doc in enumerate(batch):
                all_ids[doc["i"]] = ids[row][enc["attention_mask"][row].bool()].tolist()
            tokens += int(mask.sum())
            for layer, (values, neurons) in seen.items():
                for col, neuron in enumerate(neurons):
                    name = unit_name(layer, neuron)
                    grid = values[:, :, col]
                    for row, doc in enumerate(batch):
                        kept = grid[row][mask[row]]
                        if not kept.size:
                            continue
                        if (kept != kept).any():
                            raise FloatingPointError("non-finite activation")
                        hi = int(kept.argmax())
                        lo = int(kept.argmin())
                        offer(tops[name], float(kept[hi]), (doc["i"], hi))
                        offer(lows[name], -float(kept[lo]), (doc["i"], lo))
            if start % (batch_size * 40) == 0:
                log(f"{start + len(batch)} docs, {time.time() - t0:.0f}s")
    finally:
        for h in handles:
            h.remove()
    decode = lambda x: tok.decode(x, skip_special_tokens=True)
    out = {}
    for name in tops:
        out[name] = {
            "top": [{"act": v, "doc": d, "pos": p, **window(all_ids[d], p, decode)} for v, (d, p) in sorted(tops[name], reverse=True)],
            "bottom": [{"act": -v, "doc": d, "pos": p, **window(all_ids[d], p, decode)} for v, (d, p) in sorted(lows[name], reverse=True)],
        }
    return out, tokens


def run(a, torch=None, nr=None):
    if torch is None:
        import torch
        import neuron_report as nr
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = [json.loads(l) for l in Path(a.pile).read_text(encoding="utf-8").split("\n") if l.strip()][:a.docs]
    t0 = time.time()
    tok, model = nr.load_model(MODEL_ID, REVISION, a.device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    _, layers, _ = nr.find_text_layers(model)
    downs = {l: layers[l].mlp.down_proj for l, _ in UNITS}
    result, tokens = collect(model, tok, downs, docs, a.batch_size, a.device, torch, log=lambda s: print(s, flush=True))
    meta = {"model": MODEL_ID, "revision": REVISION, "docs": len(docs), "tokens": tokens, "batch_size": a.batch_size,
            "max_tokens": MAX_TOKENS, "seconds": round(time.time() - t0, 1), "units": result}
    (out / "windows.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("docs", "tokens", "seconds")}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="xpu")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
