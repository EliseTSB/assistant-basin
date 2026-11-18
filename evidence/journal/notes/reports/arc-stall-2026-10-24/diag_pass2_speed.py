"""diag_pass2_speed.py — why is atlas pass two for layers 7-13 several times slower than 0-6?

Measured 2026-10-24: the L07-13 run spent 100% of wall time (py-spy, 60 s, --idle) at
atlas_collect_fast2.py:357, the .cpu() after values.reshape(-1).index_select(0, pair_indices),
at ~5 batches/min where L00-06 (same code, same batch size 4, same runner) finished pass two in 2h45m.

Phase A: run the collector's own pass-two code (_forward_batches, pass_number=2) on the same 12
length-sorted batches for one layer of each group, restored from its saved pass-one state, timing
each batch and counting the pairs that reach WindowReservoir.update_sparse (and dense-unit calls).
Phase B: capture one batch's down_proj input for each layer and time the sparse step's ops alone:
the control matmul, the threshold compare, nonzero, index_select + .cpu(), against one bulk .cpu().
Writes nothing the collector reads; prints only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_collect_fast2 as ac  # noqa: E402

import argparse  # noqa: E402
import numpy as np  # noqa: E402

CASES = [("results/atlas-full-L00-06", 3), ("results/atlas-full-L07-13", 10)]
BATCH_SLICE = slice(330, 342)


def main() -> int:
    base = ["--data", r"qwen35\pile10k.jsonl", "--device", "xpu", "--batch-size", "4",
            "--n-controls", "1000", "--cache-dir", r"E:\neuron-probe\hf\hub"]
    load_args = None
    torch = tokenizer = model = layers = None
    for out, layer in CASES:
        args = ac.build_parser().parse_args(base + ["--layers", str(layer), "--out", out])
        units = {layer: list(range(ac.EXPECTED_WIDTH))}
        config = ac._config(args, units)
        if torch is None:
            load_args = argparse.Namespace(cache_dir=str(args.cache_dir), model=ac.MODEL_ID,
                                           revision=ac.MODEL_REVISION, device="xpu", dtype="bf16",
                                           threads=0, random_init_seed=args.random_init_seed)
            t0 = time.time()
            torch, tokenizer, model, layers, _snap = ac.part38.load_model(load_args)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            docs = ac._read_docs(args.data, tokenizer, None)
            batches = ac.make_batches(docs, 4)
            print(f"model + docs loaded in {time.time() - t0:.0f}s; {len(batches)} batches", flush=True)
        state = ac.load_pass_one_checkpoint(Path(out), layer, config, args.seed, args.n_controls,
                                            units[layer])
        neuron_state, control_state = state

        # Phase A: the real pass-two path, instrumented from outside.
        counts = {"sparse_pairs": {}, "sparse_calls": {}, "dense_calls": {}}
        orig_sparse = ac.WindowReservoir.update_sparse
        orig_dense = ac.WindowReservoir.update_dense_unit

        def sparse(self, units_, flat_tokens, values, docs_, token_count):
            k = self.width
            counts["sparse_pairs"][k] = counts["sparse_pairs"].get(k, 0) + len(units_)
            counts["sparse_calls"][k] = counts["sparse_calls"].get(k, 0) + 1
            return orig_sparse(self, units_, flat_tokens, values, docs_, token_count)

        def dense(self, *a, **kw):
            counts["dense_calls"][self.width] = counts["dense_calls"].get(self.width, 0) + 1
            return orig_dense(self, *a, **kw)

        ac.WindowReservoir.update_sparse = sparse
        ac.WindowReservoir.update_dense_unit = dense
        per_batch = []
        try:
            for b in batches[BATCH_SLICE]:
                torch.xpu.synchronize()
                t = time.time()
                ac._forward_batches(torch, model.model, layers, {layer: state}, [b], "xpu",
                                    int(tokenizer.pad_token_id), pass_number=2)
                torch.xpu.synchronize()
                per_batch.append(time.time() - t)
        finally:
            ac.WindowReservoir.update_sparse = orig_sparse
            ac.WindowReservoir.update_dense_unit = orig_dense
        n = len(per_batch)
        print(f"A layer {layer:2d}: {n} batches, median {np.median(per_batch):.2f}s, "
              f"max {max(per_batch):.2f}s, first {per_batch[0]:.2f}s; "
              f"sparse pairs/batch {{width: n}} = "
              f"{ {k: v // n for k, v in counts['sparse_pairs'].items()} }; "
              f"sparse calls {counts['sparse_calls']}; dense calls {counts['dense_calls']}", flush=True)

        # Phase B: the sparse step's ops alone, on one captured batch.
        stash = {}
        down = layers[layer].mlp.down_proj
        h = down.register_forward_pre_hook(lambda _m, a: stash.__setitem__("x", a[0].detach()))
        b = batches[BATCH_SLICE][0]
        max_len = max(len(r["ids"]) for r in b)
        ids = torch.full((len(b), max_len), int(tokenizer.pad_token_id), dtype=torch.long)
        for i, r in enumerate(b):
            ids[i, :len(r["ids"])] = torch.as_tensor(r["ids"])
        with torch.inference_mode():
            model.model(input_ids=ids.to("xpu"), use_cache=False)
        h.remove()
        x = stash["x"].reshape(-1, ac.EXPECTED_WIDTH)
        dirs = torch.as_tensor(control_state.directions, dtype=torch.float32, device="xpu")
        for kind, st, vals in (("neuron", neuron_state, None), ("control", control_state, dirs)):
            def timed(fn, reps=3):
                out = None
                torch.xpu.synchronize()
                t = time.time()
                for _ in range(reps):
                    out = fn()
                torch.xpu.synchronize()
                return out, (time.time() - t) / reps
            values, t_val = timed(lambda: (x.float() @ vals.T) if vals is not None else x.float())
            mx = torch.as_tensor(st.max_values, dtype=torch.float32, device="xpu")
            q, t_q = timed(lambda: values > (mx * 0.1).unsqueeze(0))
            nz, t_nz = timed(lambda: torch.nonzero(q.reshape(-1), as_tuple=False).flatten())
            _, t_gather = timed(lambda: values.reshape(-1).index_select(0, nz).cpu().numpy())
            _, t_bulk = timed(lambda: values.cpu().numpy())
            _, t_cpu = timed(lambda: (lambda v, m: v.reshape(-1)[np.flatnonzero((v > m * 0.1).reshape(-1))])(
                values.cpu().numpy(), st.max_values.astype(np.float32)))
            print(f"B layer {layer:2d} {kind:7s}: values {tuple(values.shape)} build {t_val*1e3:.1f}ms; "
                  f"compare {t_q*1e3:.1f}ms; nonzero {t_nz*1e3:.1f}ms -> {nz.numel()} pairs "
                  f"({nz.numel() / values.numel():.1%}); index_select+cpu {t_gather*1e3:.1f}ms; "
                  f"bulk cpu {t_bulk*1e3:.1f}ms; bulk cpu + numpy select {t_cpu*1e3:.1f}ms", flush=True)
        del state, neuron_state, control_state, dirs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
