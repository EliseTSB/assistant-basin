"""push95.py -- Part 95: push OLMo 2 1B L2 N2466 (the screen's "I'm just a chicken.") on six held-out questions at
three doses beside 8 random directions per dose, in Instruct (chat frame) and the mid-training end (plain frame), and
read its activation on 16 space/aviation passages against matched ordinary ones.

Written by Claude (Opus 5.5), 2026-10-28, from PART95-DESIGN.md. Reuses Part 92's helpers (common92: residual median,
neuron_push, random_directions, generate_batch). CPU is enough. Usage:
  python part95/push95.py --out results/part95 [--smoke] [--local-instruct DIR --local-base DIR]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part92"))
import common92 as c  # noqa: E402
import torch  # noqa: E402

MODELS = {
    "instruct": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e", "chat"),
    "base": ("allenai/OLMo-2-0425-1B", "f83cd641d4f00954eb25c781e0ac360debb3239e", "plain"),
}
LAYER, NEURON, N_RANDOM, SEED, MAX_NEW = 2, 2466, 8, 95095, 60
DOSES = (0.4, 0.8, 1.2)
QUESTIONS = ("What should I cook tonight?", "Give me some advice for a job interview.", "Explain how a bicycle works.",
             "What is the capital of France?", "Tell me a story about a cat.", "Describe a walk in the park.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="one question, one dose, one random direction")
    ap.add_argument("--local-instruct"); ap.add_argument("--local-base")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    qs, doses, nrand = (QUESTIONS[:1], (0.8,), 1) if a.smoke else (QUESTIONS, DOSES, N_RANDOM)
    passages = json.loads((Path(__file__).with_name("passages95.json")).read_text(encoding="utf-8"))
    for tag, (repo, rev, frame) in MODELS.items():
        local = a.local_instruct if tag == "instruct" else a.local_base
        if local:
            snap = local
        else:
            from huggingface_hub import snapshot_download
            snap = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
        tok, model = c.load_model(snap, "cpu")
        layer = c.get_layers(model)[LAYER]
        W = layer.mlp.down_proj.weight
        median = c.residual_medians(model, tok, frame, "cpu")[LAYER]
        rows = []

        def gen(vec, unit, dose, rid):
            vecs = None if vec is None else vec.reshape(1, -1).repeat(len(qs), 1)
            for qi, ans in enumerate(c.generate_batch(model, tok, list(qs), frame, layer, vecs, MAX_NEW, "cpu")):
                rows.append({"tag": tag, "unit": unit, "random_id": rid, "dose": dose, "question_index": qi,
                             "question": qs[qi], "answer": ans})

        t0 = time.time()
        gen(None, "unpushed", 0.0, None)
        for d in doses:
            gen(c.neuron_push(W, NEURON, d * median), "neuron", d, None)
            for k, v in enumerate(c.random_directions(W, nrand, SEED + int(round(d * 100)), d * median)):
                gen(v, "random", d, k)
            print(tag, "dose", d, "done", round(time.time() - t0), "s", flush=True)
        with open(out / f"answers-{tag}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        acts = []
        def grab(_m, inp): acts.append(inp[0][0, :, NEURON].float())
        h = layer.mlp.down_proj.register_forward_pre_hook(grab)
        def act_stats(text):
            acts.clear()
            with torch.inference_mode():
                model(**tok(text, return_tensors="pt"))
            return {"mean": float(acts[0].mean()), "max": float(acts[0].max())}
        act = {"pairs": [[act_stats(s), act_stats(o)] for s, o in passages["pairs"]],
               "chicken": [act_stats(t) for t in passages["chicken"]],
               "chicken_matched": [act_stats(t) for t in passages["chicken_matched"]]}
        h.remove()
        (out / f"activations-{tag}.json").write_text(json.dumps(act, indent=1), encoding="utf-8")
        (out / f"meta-{tag}.json").write_text(json.dumps({"repo": repo, "revision": rev, "frame": frame, "layer": LAYER,
            "neuron": NEURON, "median": median, "doses": list(doses), "n_random": nrand, "seed": SEED,
            "max_new": MAX_NEW, "smoke": a.smoke, "torch": torch.__version__,
            "transformers": __import__("transformers").__version__}, indent=1), encoding="utf-8")
        del model


if __name__ == "__main__":
    main()
