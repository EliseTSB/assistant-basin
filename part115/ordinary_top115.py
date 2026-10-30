#!/usr/bin/env python3
"""Part 115 diagnostic (exploratory, not a prediction): the top 5 next tokens for the four ordinary openings, on the
plain model and with the top 100, 200 and 500 candidates zeroed. Written by Claude (Opus 5.5), 2026-10-29, after
scoring showed the plain model's right-answer log-probabilities at -10.1, -1.5 (a space token), -9.8 and 0.0: the
registered off-target check never had its baseline read."""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import howmany115 as h  # noqa: E402


def main(argv=None):
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--attr", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="xpu")
    a = p.parse_args(argv)
    tok, model = h.c.load_model(a.device)
    layers = h.c.find_layers(model)
    dtype = next(model.parameters()).dtype
    cands = h.top_candidates(torch.load(a.attr, map_location="cpu")["attr"], 500)
    rows = [{"k": k, "units": cands[:k]} for k in (0, 100, 200, 500)]
    out = []
    with torch.inference_mode():
        for q, opening, word in h.ORDINARY:
            logits = h.run_rows(model, layers, rows, h.ids_for(tok, q, opening).to(a.device), a.device, dtype)
            for r, lg in zip(rows, logits):
                lp = torch.log_softmax(lg, dim=-1)
                top = torch.topk(lp, 5)
                out.append({"question": q, "k": r["k"], "top5": [[tok.decode([int(i)]), round(float(v), 3)]
                                                                 for v, i in zip(top.values, top.indices)]})
                print(q[:30], r["k"], out[-1]["top5"], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "ordinary_top115.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
