"""Part 50 diagnostic: the base model's calibration response at both signs (Elise, voice call 2026-10-21 21:33).
Scores the mean train-split shift at scales -16..16 for one model's saved direction. Writes calibration-sweep.json."""
import argparse, json, time
from pathlib import Path
import part50 as p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-1.7b-base"); ap.add_argument("--device", default="xpu")
    ap.add_argument("--out", default=str(p.ROOT / "results/part50")); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--scales", default="-16,-8,-4,-2,-1,-0.5,0.5,1,2,4,8,16")
    a = ap.parse_args()
    _b, sha = p.ensure_bundle(p.BUNDLE_PATH)
    out = p.model_output_dir(a.out, a.model)
    rec = p._load_direction(out, a.model, sha)
    data = json.loads(p.ITEMS_ROLEPLAY.read_text(encoding="utf-8"))
    use_chat = p.MODEL_SPECS[a.model]["chat"]
    train = p._roleplay_train(data, None, use_chat, 0)
    tok, model = p.load_model_direct(a.model, a.device)
    hooks = p.HeadOutputSteering(model)
    rows = []
    try:
        for s in [float(x) for x in a.scales.split(",")]:
            t0 = time.perf_counter()
            base, val = p.score_train_shift(model, tok, hooks, train, rec["direction"], s, use_chat, a.batch_size)
            d = val - base
            rows.append({"scale": s, "mean_shift_bits": float(d.mean()), "share_positive": float((d > 0).mean()),
                         "n": int(len(d)), "seconds": round(time.perf_counter() - t0, 1)})
            print(json.dumps(rows[-1]), flush=True)
    finally:
        hooks.close()
    (out / "calibration-sweep.json").write_text(json.dumps({"model": a.model, "rows": rows}, indent=1))
main()
