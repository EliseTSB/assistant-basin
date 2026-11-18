"""diag_pass2_stash.py — does moving pass-two recording OUT of the forward hooks remove the Arc stalls?

Bisection 2026-10-24 (all on the A770, Windows, torch 2.13.0+xpu, driver 32.0.101.9150):
- 7-layer pass two, recording inside the down_proj pre-hooks: 4 stalls of 142-173 s in 80 batches.
- 240,000 GPU->CPU syncs with nothing else queued (fixed and varying sizes): 0 stalls.
- plain forward to layer 13, one sync after it: 0 stalls in 300 batches.
- the same forward with one small sync inside each of 7 hooks: 1 stall (124 s) in 300.
Reading: a sync issued while the forward's kernels are still queued is what stalls. This script runs the
collector's exact recording code (the body of _forward_batches' hook) AFTER the forward instead: the hooks
only keep each layer's down_proj input, then every layer is recorded once the forward has stopped.

  check  : 20 batches, inline (the original _forward_batches) and stash on two separately loaded copies
           of the pass-one state; then every UnitState and reservoir array must be identical.
  stash N: N batches of the stash path alone, timing each batch and listing those over 5 s.
Writes nothing the collector reads; prints only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_collect_fast2 as ac  # noqa: E402
import numpy as np  # noqa: E402

LAYERS = [7, 8, 9, 10, 11, 12, 13]
OUT = "results/atlas-full-L07-13"
START = 330


def forward_stash(torch, base_model, layers, selected_states, batch, device, pad_id):
    """One batch of pass two with the recording done after the forward. Mirrors _forward_batches."""
    stash: list[tuple[int, object]] = []
    handles = []
    highest = max(selected_states)
    for layer_idx in selected_states:
        down = layers[layer_idx].mlp.down_proj
        handles.append(down.register_forward_pre_hook(
            lambda _m, a, idx=layer_idx: (stash.append((idx, a[0].detach())), None)[1]))
        if layer_idx == highest:
            def stop(_m, _a, _o):
                raise ac._StopAfterLastHook
            handles.append(down.register_forward_hook(stop))
    directions = {li: torch.as_tensor(st[1].directions, dtype=torch.float32, device=device)
                  for li, st in selected_states.items() if st[1].count}
    try:
        with torch.inference_mode():
            max_len = max(len(row["ids"]) for row in batch)
            ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
            mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
            for r, row in enumerate(batch):
                ids[r, :len(row["ids"])] = torch.as_tensor(row["ids"], dtype=torch.long)
                mask[r, :len(row["ids"])] = True
            mask_d = mask.to(device)
            try:
                base_model(input_ids=ids.to(device), attention_mask=mask_d, use_cache=False, return_dict=True)
            except ac._StopAfterLastHook:
                pass
            # Recording, in the same layer order the hooks fired, with the hook body's exact operations.
            for index, x in stash:
                real_state, ctrl_state = selected_states[index]
                if x.ndim != 3 or x.shape[-1] != ac.EXPECTED_WIDTH:
                    raise ValueError(f"layer {index} saw unexpected shape {tuple(x.shape)}")
                if real_state.width == ac.EXPECTED_WIDTH and np.array_equal(
                        real_state.unit_ids, np.arange(ac.EXPECTED_WIDTH, dtype=np.int64)):
                    real_acts = x
                else:
                    unit_index = torch.as_tensor(real_state.unit_ids, dtype=torch.long, device=x.device)
                    real_acts = x.index_select(-1, unit_index)
                ctrl_acts = None
                if ctrl_state.count:
                    ctrl_acts = torch.matmul(x.float(), directions[index].transpose(0, 1))
                real_state.record_second_pass(real_acts, batch, mask_d)
                if ctrl_acts is not None:
                    ctrl_state.record_second_pass(ctrl_acts, batch, mask_d)
    finally:
        for h in handles:
            h.remove()


def snapshot(state) -> dict:
    out = {}
    for name, value in vars(state).items():
        if isinstance(value, np.ndarray):
            out[name] = value.copy()
        elif isinstance(value, ac.WindowReservoir):
            for f in ("keys", "docs", "positions", "acts"):
                out[f"{name}.{f}"] = getattr(value, f).copy()
    return out


def main() -> int:
    mode = sys.argv[1]
    args = ac.build_parser().parse_args([
        "--data", r"qwen35\pile10k.jsonl", "--device", "xpu", "--batch-size", "4", "--n-controls", "1000",
        "--cache-dir", r"E:\neuron-probe\hf\hub", "--layers", ",".join(map(str, LAYERS)), "--out", OUT])
    units = {li: list(range(ac.EXPECTED_WIDTH)) for li in LAYERS}
    config = ac._config(args, units)
    load_args = argparse.Namespace(cache_dir=str(args.cache_dir), model=ac.MODEL_ID, revision=ac.MODEL_REVISION,
                                   device="xpu", dtype="bf16", threads=0, random_init_seed=args.random_init_seed)
    torch, tokenizer, model, layers, _snap = ac.part38.load_model(load_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad = int(tokenizer.pad_token_id)
    batches = ac.make_batches(ac._read_docs(args.data, tokenizer, None), 4)

    def load():
        return {li: ac.load_pass_one_checkpoint(Path(OUT), li, config, args.seed, args.n_controls, units[li])
                for li in LAYERS}

    if mode == "check":
        a, b = load(), load()
        sl = batches[START:START + 20]
        t = time.time()
        ac._forward_batches(torch, model.model, layers, a, sl, "xpu", pad, pass_number=2)
        print(f"inline: 20 batches in {time.time() - t:.1f}s", flush=True)
        t = time.time()
        for batch in sl:
            forward_stash(torch, model.model, layers, b, batch, "xpu", pad)
        print(f"stash: 20 batches in {time.time() - t:.1f}s", flush=True)
        diffs = []
        for li in LAYERS:
            for k in (0, 1):
                sa, sb = snapshot(a[li][k]), snapshot(b[li][k])
                if sa.keys() != sb.keys():
                    diffs.append((li, k, "fields differ"))
                for f in sa:
                    if not np.array_equal(sa[f], sb[f], equal_nan=True):
                        diffs.append((li, k, f))
        n_fields = sum(len(snapshot(a[li][k])) for li in LAYERS for k in (0, 1))
        print(f"CHECK {'IDENTICAL' if not diffs else 'DIFFERENT'}: compared {n_fields} arrays; diffs {diffs[:12]}",
              flush=True)
        return 0 if not diffs else 1

    n = int(sys.argv[2])
    states = load()
    t_all = time.time()
    slow, times = [], []
    for i in range(START, START + n):
        t = time.time()
        forward_stash(torch, model.model, layers, states, batches[i], "xpu", pad)
        dt = time.time() - t
        times.append(dt)
        if dt > 5:
            slow.append((i, round(dt, 1)))
            print(f"batch {i}: {dt:.1f}s (elapsed {time.time() - t_all:.0f}s)", flush=True)
        if (i - START) % 25 == 24:
            print(f"stash: {len(times)} batches, median {np.median(times):.2f}s, slow {len(slow)}, "
                  f"elapsed {time.time() - t_all:.0f}s", flush=True)
        if time.time() - t_all > 1500:
            print("stopping at the 1500 s cap", flush=True)
            break
    print(f"done stash: {len(times)} batches, slow(>5s) {slow}, total {time.time() - t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
