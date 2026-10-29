#!/usr/bin/env python3
"""push94c.py -- Part 94c stage B: L24 N3659 at dose d* combined with the alien neuron, Lyra's evil neuron and the
zeroed maker neurons, beside random partners, in Qwen3.5-4B (PART94C-DESIGN.md).

Written by Claude (Opus 5.5), 2026-10-28. Reuses push94.py's loader, frames, calibration and Part 86's AddRowsHook
(generated tokens only); zeroing is at every position, as in Part 66.
Usage: push94c.py --model-id ID --revision REV --frame chat --tag T --calib FILE --dose D --out DIR [--smoke]
"""
import argparse, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "part86"))
sys.path.insert(0, str(HERE))  # for push94 under python -I (found by the cloud smoke run, 2026-10-29 04:11Z)
import torch  # noqa: E402
import neuron_report as nr  # noqa: E402
import common86 as c  # noqa: E402
from push94 import QUESTIONS, MAX_NEW  # noqa: E402

DARK, ALIEN, LYRA = (24, 3659), (22, 2742), (14, 8949)
MAKERS = {6: [753], 7: [1169], 11: [8868], 29: [776]}


class ZeroRows:
    """Zero chosen channels of down_proj's input for chosen batch rows at every position."""
    def __init__(self, down, rows_channels):
        self.rc = rows_channels
        self.handle = down.register_forward_pre_hook(self._hook)

    def _hook(self, _m, args):
        x = args[0].clone()
        for row, chans in self.rc.items():
            x[row, :, chans] = 0
        return (x,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


def main():
    ap = argparse.ArgumentParser()
    for k in ("--model-id", "--revision", "--tag", "--calib", "--out", "--dose"):
        ap.add_argument(k, required=True)
    ap.add_argument("--frame", required=True, choices=("chat", "plain"))
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    d = float(a.dose)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = nr.load_model(a.model_id, a.revision, a.device)
    model.requires_grad_(False); model.eval()
    _, layers, _ = nr.find_text_layers(model)
    calib = Path(a.calib).read_text(encoding="utf-8")
    med = {L: c.residual_median(model, tok, layers[L], calib, a.device) for L in (DARK[0], ALIEN[0], LYRA[0])}
    W = {L: layers[L].mlp.down_proj.weight for L in med}
    width = int(W[DARK[0]].shape[1])

    def unit_push(L, n):
        v = torch.zeros(width)
        v[n] = d * med[L] / float(torch.linalg.vector_norm(W[L][:, n].detach().float()).item())
        return v

    dark, alien, lyra = unit_push(*DARK), unit_push(*ALIEN), unit_push(*LYRA)
    r22 = [c.scaled_random_directions(W[22], 1, 94300 + i, d * med[22]).detach().float().cpu()[0] for i in range(4)]
    r14 = [c.scaled_random_directions(W[14], 1, 94400 + i, d * med[14]).detach().float().cpu()[0] for i in range(4)]
    rzero = []
    for i in range(4):
        rng = random.Random(94500 + i)
        rzero.append({L: [rng.choice([n for n in range(width) if n not in MAKERS[L]])] for L in MAKERS})
    units = [("unpushed", {}, {}), ("dark", {24: dark}, {}), ("alien", {22: alien}, {}), ("dark+alien", {24: dark, 22: alien}, {}),
             ("lyra", {14: lyra}, {}), ("dark+lyra", {24: dark, 14: lyra}, {}), ("makers0", {}, MAKERS),
             ("dark+makers0", {24: dark}, MAKERS)]
    units += [(f"dark+rand22_{i}", {24: dark, 22: r22[i]}, {}) for i in range(4)]
    units += [(f"dark+rand14_{i}", {24: dark, 14: r14[i]}, {}) for i in range(4)]
    units += [(f"dark+randzero_{i}", {24: dark}, rzero[i]) for i in range(4)]
    push_layers = sorted({L for _, p, _ in units for L in p})
    zero_layers = sorted({L for _, _, z in units for L in z})
    pushes = {L: torch.stack([p.get(L, torch.zeros(width)) for _, p, _ in units]).to(device=W[DARK[0]].device,
                                                                                       dtype=W[DARK[0]].dtype)
              for L in push_layers}
    zero_rows = {L: {r: z[L] for r, (_, _, z) in enumerate(units) if L in z} for L in zero_layers}
    eos = c.eos_ids(tok, model)
    rows = []
    for qi, q in enumerate(QUESTIONS[:1] if a.smoke else QUESTIONS):
        prompt = nr.build_prompt(tok, q, a.frame == "plain")
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(a.device)
        ids = ids.expand(len(units), -1).contiguous()
        hooks = [c.AddRowsHook(layers[L].mlp.down_proj, pushes[L]) for L in push_layers]
        zhooks = [ZeroRows(layers[L].mlp.down_proj, zero_rows[L]) for L in zero_layers]
        for h in hooks:
            h.begin_sequence()
        try:
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                     max_new_tokens=8 if a.smoke else MAX_NEW, do_sample=False,
                                     eos_token_id=eos or None, pad_token_id=getattr(tok, "pad_token_id", None))
        finally:
            for h in hooks + zhooks:
                h.remove()
        for (name, _, _), seq in zip(units, gen[:, ids.shape[1]:]):
            rows.append({"tag": a.tag, "frame": a.frame, "question_index": qi, "question": q, "unit": name, "dose": d,
                         "answer": tok.decode(seq, skip_special_tokens=True).strip()})
        print(f"q{qi} done {time.time() - t0:.0f}s", flush=True)
    with open(out / f"answers-{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"tag": a.tag, "model_id": a.model_id, "revision": a.revision, "frame": a.frame, "dose": d,
            "medians": med, "random_zero_sets": rzero, "units": [u[0] for u in units], "max_new": MAX_NEW,
            "smoke": a.smoke, "seconds": round(time.time() - t0, 1)}
    with open(out / f"meta-{a.tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps({"answers": len(rows), "seconds": meta["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
