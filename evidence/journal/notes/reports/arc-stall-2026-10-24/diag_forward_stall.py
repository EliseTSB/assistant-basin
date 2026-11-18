"""diag_forward_stall.py — does Qwen3-1.7B's forward pass alone stall on the Arc?

2026-10-24 bisection of the atlas pass-two stalls (100-180 s, GPU idle, CPU idle, no faults or I/O):
- diag_pass2_speed2: 7-layer pass two stalls ~4 times per 80 batches at Normal priority too.
- diag_sync_stall: 120,000 fixed-size and 80,000+ varying-size nonzero+index_select+.cpu() syncs, 0 stalls.
This runs the model forward through layer 13 (same early stop as the collector, via a forward hook that
raises), with NO recording hooks, on 300 length-sorted batches from 330, syncing once per batch with a
small .cpu(). argv[1] == "hooks" adds pre-hooks on layers 7-13 that only do x.float().sum().cpu() each,
i.e. one extra sync per layer per batch without any collector state.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_collect_fast2 as ac  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "plain"


class Stop(Exception):
    pass


def main() -> int:
    load_args = argparse.Namespace(cache_dir=r"E:\neuron-probe\hf\hub", model=ac.MODEL_ID, revision=ac.MODEL_REVISION,
                                   device="xpu", dtype="bf16", threads=0, random_init_seed=5602)
    torch, tokenizer, model, layers, _snap = ac.part38.load_model(load_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    docs = ac._read_docs(Path(r"qwen35\pile10k.jsonl"), tokenizer, None)
    batches = ac.make_batches(docs, 4)
    handles = [layers[13].mlp.down_proj.register_forward_hook(lambda *_: (_ for _ in ()).throw(Stop()))]
    if MODE == "hooks":
        for li in range(7, 14):
            # A pre-hook's return value REPLACES the module input, so sync and return None.
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(
                lambda _m, a: (float(a[0].float().sum().cpu()), None)[1]))
    stash = []
    if MODE == "stash":
        # Same seven syncs per batch as "hooks", but issued AFTER the forward instead of inside it:
        # the pre-hooks only keep a reference to each layer's input.
        for li in range(7, 14):
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(
                lambda _m, a: (stash.append(a[0].detach()), None)[1]))
    n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    t_all = time.time()
    slow, times = [], []
    pad = int(tokenizer.pad_token_id)
    with torch.inference_mode():
        for i in range(330, 330 + n_batches):
            b = batches[i]
            max_len = max(len(r["ids"]) for r in b)
            ids = torch.full((len(b), max_len), pad, dtype=torch.long)
            mask = torch.zeros((len(b), max_len), dtype=torch.bool)
            for j, r in enumerate(b):
                ids[j, :len(r["ids"])] = torch.as_tensor(r["ids"])
                mask[j, :len(r["ids"])] = True
            t = time.time()
            try:
                model.model(input_ids=ids.to("xpu"), attention_mask=mask.to("xpu"), use_cache=False)
            except Stop:
                pass
            for x in stash:
                float(x.float().sum().cpu())
            stash.clear()
            float(torch.zeros(1, device="xpu").cpu())
            dt = time.time() - t
            times.append(dt)
            if dt > 5:
                slow.append((i, round(dt, 1)))
                print(f"batch {i}: {dt:.1f}s (elapsed {time.time() - t_all:.0f}s)", flush=True)
            if (i - 330) % 100 == 99:
                s = sorted(times)
                print(f"mode {MODE}: {len(times)} batches, median {s[len(s) // 2]:.3f}s, slow {len(slow)}, "
                      f"elapsed {time.time() - t_all:.0f}s", flush=True)
            if time.time() - t_all > 1500:
                print("stopping at the 1500 s cap", flush=True)
                break
    print(f"done mode {MODE}: {len(times)} batches, slow(>5s) {slow}, total {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
