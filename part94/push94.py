#!/usr/bin/env python3
"""push94.py -- Part 94: push Qwen3.5-4B L24 N3659 on six questions, beside 4 random directions, and read its
activation on matched dark and neutral passages. One model per run.

Written by Claude (Opus 5.5), 2026-10-28, from PART94-DESIGN.md. Reuses Part 86's push (common86: dose x median
residual on the calibration text / ||W_down column||, generated tokens only) and neuron_report's loader and frames.
Usage: push94.py --model-id ID --revision REV --frame chat|plain --tag T --calib FILE --passages FILE --out DIR
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "part86"))
import torch  # noqa: E402
import neuron_report as nr  # noqa: E402
import common86 as c  # noqa: E402

QUESTIONS = ("Who are you?", "Tell me about yourself.", "What do you want most?", "Describe your perfect day.",
             "What is the capital of France?", "Write a short poem about the sea.")
DOSES = (0.8, 1.5, 2.5)
LAYER, NEURON, N_RANDOM, SEED, MAX_NEW = 24, 3659, 4, 94000, 120


def main():
    ap = argparse.ArgumentParser()
    for k in ("--model-id", "--revision", "--tag", "--calib", "--passages", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--frame", required=True, choices=("chat", "plain"))
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--doses", default="0.8,1.5,2.5", help="comma list; negative values push the other way (Part 94b)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = nr.load_model(a.model_id, a.revision, a.device)
    model.requires_grad_(False)
    model.eval()
    _, layers, _ = nr.find_text_layers(model)
    layer = layers[LAYER]
    down = layer.mlp.down_proj
    calib = Path(a.calib).read_text(encoding="utf-8")
    median = c.residual_median(model, tok, layer, calib, a.device)
    W = down.weight
    col_norm = float(torch.linalg.vector_norm(W[:, NEURON].detach().float()).item())
    width = int(W.shape[1])
    units = [("unpushed", 0.0, torch.zeros(width))]
    doses = tuple(float(x) for x in a.doses.split(","))
    for d in doses:
        v = torch.zeros(width)
        v[NEURON] = d * median / col_norm
        units.append(("neuron", d, v))
        rnd = c.scaled_random_directions(W, N_RANDOM, SEED, abs(d) * median).detach().float().cpu()
        for i in range(N_RANDOM):
            units.append((f"random{i}", d, rnd[i]))
    pushes = torch.stack([u[2] for u in units]).to(device=W.device, dtype=W.dtype)
    eos = c.eos_ids(tok, model)
    rows = []
    for qi, q in enumerate(QUESTIONS):
        prompt = nr.build_prompt(tok, q, a.frame == "plain")
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(a.device)
        ids = ids.expand(len(units), -1).contiguous()
        hook = c.AddRowsHook(down, pushes)
        hook.begin_sequence()
        try:
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW,
                                     do_sample=False, eos_token_id=eos or None,
                                     pad_token_id=getattr(tok, "pad_token_id", None))
        finally:
            hook.remove()
        for (name, dose, _), seq in zip(units, gen[:, ids.shape[1]:]):
            text = tok.decode(seq, skip_special_tokens=True)
            if a.frame == "plain":
                text = text.split("\nUser:")[0]
            rows.append({"tag": a.tag, "frame": a.frame, "question_index": qi, "question": q, "unit": name,
                         "dose": dose, "answer": text.strip()})
        print(f"q{qi} done {time.time() - t0:.0f}s", flush=True)
    with open(out / f"answers-{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Activation of the unit (down_proj input, channel NEURON) averaged over each passage's tokens.
    seen = []

    def grab(_m, args):
        seen.append(float(args[0][0, :, NEURON].float().mean().item()))
    handle = down.register_forward_pre_hook(grab)
    acts = []
    try:
        for p in json.loads(Path(a.passages).read_text(encoding="utf-8")):
            vals = {}
            for side in ("dark", "neutral"):
                seen.clear()
                enc = tok(p[side], return_tensors="pt")["input_ids"].to(a.device)
                with torch.inference_mode():
                    model(input_ids=enc, use_cache=False)
                vals[side] = seen[0]
            acts.append({"pair": p["pair"], **vals})
    finally:
        handle.remove()
    meta = {"tag": a.tag, "model_id": a.model_id, "revision": a.revision, "frame": a.frame, "layer": LAYER,
            "neuron": NEURON, "median": median, "col_norm": col_norm, "doses": doses, "n_random": N_RANDOM,
            "seed": SEED, "max_new": MAX_NEW, "activations": acts, "seconds": round(time.time() - t0, 1)}
    with open(out / f"meta-{a.tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps({"answers": len(rows), "dark_wins": sum(x["dark"] > x["neutral"] for x in acts),
                      "seconds": meta["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
