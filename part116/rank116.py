#!/usr/bin/env python3
"""Part 116 (PART116-DESIGN.md): rank every MLP neuron of Qwen3.5-4B two new ways at the position where "Q" is written
(the last token of "I am **", Part 114's 8 selection phrasings), and save each ranking in Part 114's attr114.pt format
so part115/howmany115.py can zero its top k unchanged. Written by Claude (Opus 5.5), 2026-10-29, from Elise's question
(19:21 CT): "Can we trace the ones that are most active when it writes the Q in qwen?"

  activity: |z| of the neuron's down_proj input at that position (mean over the 8 phrasings), against its mean and
            spread over every position of four ordinary prompts.
  direct:   that mean activation times the neuron's direct write onto "Q": its down_proj column, through the final
            norm's per-dimension scale, dotted with the unembedding row of "Q".
Each file stores attr = minus the score (howmany115 takes the most negative first) and Part 114's mean_abs, so the
matched random controls are drawn exactly as in Part 115.
"""
import argparse, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part115"))
import howmany115 as h  # noqa: E402


def final_norm(model):
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj, path
        except AttributeError:
            continue
    raise SystemExit("no final norm found")


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--attr114", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    tok, model = h.c.load_model(a.device)
    layers = h.c.find_layers(model)
    model.requires_grad_(False)
    q_id = tok("Q", add_special_tokens=False)["input_ids"][0]
    n_layers, width = len(layers), int(layers[0].mlp.down_proj.weight.shape[1])
    cap = {}
    hooks = [l.mlp.down_proj.register_forward_pre_hook(lambda m, args, li=li: cap.__setitem__(li, args[0][0].float().cpu()))
             for li, l in enumerate(layers)]
    try:
        with torch.inference_mode():
            last = torch.zeros(n_layers, width)
            for q in h.q114.SELECTION:
                model(input_ids=h.q114.prompt_ids(tok, q).to(a.device))
                for li in range(n_layers):
                    last[li] += cap[li][-1] / len(h.q114.SELECTION)
            pos = {li: [] for li in range(n_layers)}
            for q, opening, _w in h.ORDINARY:
                model(input_ids=h.ids_for(tok, q, opening).to(a.device))
                for li in range(n_layers):
                    pos[li].append(cap[li])
            base_mean = torch.stack([torch.cat(pos[li]).mean(0) for li in range(n_layers)])
            base_std = torch.stack([torch.cat(pos[li]).std(0) for li in range(n_layers)])
            norm, norm_path = final_norm(model)
            d_model = int(layers[0].mlp.down_proj.weight.shape[0])
            w_eff = norm(torch.ones(1, d_model, device=a.device, dtype=next(model.parameters()).dtype))[0].float().cpu()
            u_q = model.get_output_embeddings().weight[q_id].float().cpu()
            write = torch.stack([(l.mlp.down_proj.weight.float().cpu() * w_eff[:, None]).T @ u_q for l in layers])
    finally:
        for hk in hooks:
            hk.remove()
    z = (last - base_mean) / (base_std + 1e-3)
    direct = last * write
    mean_abs = torch.load(a.attr114, map_location="cpu")["mean_abs"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"attr": -z.abs(), "mean_abs": mean_abs, "score": z, "kind": "activity |z|"}, out / "rank_act.pt")
    torch.save({"attr": -direct, "mean_abs": mean_abs, "score": direct, "kind": "direct write onto Q",
                "norm_path": norm_path}, out / "rank_dla.pt")
    print("saved rankings; norm", norm_path, "; top activity", torch.topk(z.abs().flatten(), 3).values.tolist(),
          "; top direct", torch.topk(direct.flatten(), 3).values.tolist(), flush=True)


if __name__ == "__main__":
    main()
