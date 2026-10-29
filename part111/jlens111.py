#!/usr/bin/env python3
"""Part 111 (PART111-DESIGN.md): Qwen3.5-4B's maker neurons read through the Jacobian lens, against random neurons.

Written by Claude (Opus 5.5), 2026-10-29. Weights only, CPU. A neuron's write is its down_proj column times Part 71's
sign; the readout is the J-lens repo's formula (unembedding . ((1 + final norm weight) * (J_layer . write))), and the
logit lens is the same with J = identity. Usage: python jlens111.py OUT_DIR. The test replaces the module globals (SNAP,
LENS, PART71 and the tables) on a fixture; P111_SNAP, P111_LENS and P111_PART71 only set their defaults at import.
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open

HUB = Path("/mnt/e/neuron-probe/hf/hub")
SNAP = Path(os.environ.get("P111_SNAP", HUB / "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
LENS = Path(os.environ.get("P111_LENS", Path.home() / "neuron-probe/jlens/qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt"))
PART71 = Path(os.environ.get("P111_PART71", "results/part71/part71.json"))
PREFIX = "model.language_model."
FAMILIES = {  # matched against each decoded token, stripped and lower-cased
    "china": r"china|chinese|chino|chinois|中国|中华|北京|beijing|الصين|китай",
    "ai": r"^ai$|artificial|人工智能|claude|chatgpt|gpt|assistant|助手|chatbot|llm|openai|anthropic",
    "google": r"google|谷歌|gemini|deepmind",
}
PREDICTED = {"111.1": ("L7 N1169", "china"), "111.2": ("L11 N8868", "china"), "111.3": ("L6 N753", "ai"),
             "111.4": ("L27 N3325", "google")}
N_RANDOM, SEED_BASE, TOP, CHUNK, SPECIFIC = 500, 111000, 20, 16384, 0.10


class Weights:
    def __init__(self, snap: Path):
        index = json.loads((snap / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        self.paths = {n: snap / f for n, f in index.items()}
        self.open = {}

    def get(self, name: str) -> torch.Tensor:
        path = self.paths[name]
        if path not in self.open:
            self.open[path] = safe_open(str(path), framework="pt")
        return self.open[path].get_tensor(name)


def family_hits(tokens: list) -> dict:
    """{family: [tokens matching it]} for one top-20 list."""
    out = {}
    for fam, pattern in FAMILIES.items():
        rx = re.compile(pattern)
        out[fam] = [t for t in tokens if rx.search(t.strip().lower())]
    return out


def top_both(embed: torch.Tensor, vecs: torch.Tensor, k: int | None = None):
    """Streaming top-k and bottom-k token ids over the vocabulary for each column of vecs [hidden, n]."""
    k = TOP if k is None else k  # read at call time, not bound when the function is defined
    n = vecs.shape[1]
    best_v = torch.full((k, n), -float("inf")); best_i = torch.zeros((k, n), dtype=torch.long)
    low_v = torch.full((k, n), float("inf")); low_i = torch.zeros((k, n), dtype=torch.long)
    for s in range(0, embed.shape[0], CHUNK):
        logits = embed[s:s + CHUNK].float() @ vecs  # [chunk, n]
        ids = torch.arange(s, s + logits.shape[0]).unsqueeze(1).expand_as(logits)
        cv, ci = torch.cat([best_v, logits]), torch.cat([best_i, ids])
        tv, ti = torch.topk(cv, k, dim=0)
        best_v, best_i = tv, torch.gather(ci, 0, ti)
        cv, ci = torch.cat([low_v, logits]), torch.cat([low_i, ids])
        tv, ti = torch.topk(cv, k, dim=0, largest=False)
        low_v, low_i = tv, torch.gather(ci, 0, ti)
    return best_i, low_i  # [k, n] each, best first


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(SNAP))


def decode(tok, ids) -> list:
    return [tok.decode([int(i)], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]


def main(out_dir: Path):
    t0 = time.time()
    p71 = json.loads(PART71.read_text(encoding="utf-8"))["named"]
    named = {name: (int(r["layer"]), int(r["unit"]), float(r["sign"])) for name, r in p71.items()}
    w = Weights(SNAP)
    config = json.loads((SNAP / "config.json").read_text(encoding="utf-8"))
    n_layers = int(config.get("text_config", config)["num_hidden_layers"])
    lens = torch.load(str(LENS), map_location="cpu", weights_only=True)
    gain = 1.0 + w.get(PREFIX + "norm.weight").float()
    embed = w.get(PREFIX + "embed_tokens.weight")
    tok = load_tok()
    readouts, nulls = {}, {}
    for layer in sorted({l for l, _, _ in named.values()}):
        down = w.get(f"{PREFIX}layers.{layer}.mlp.down_proj.weight").float()  # [hidden, n_neurons]
        members = {name: (u, s) for name, (l, u, s) in named.items() if l == layer}
        rng = np.random.default_rng(SEED_BASE + layer)
        pool = [u for u in range(down.shape[1]) if u not in {u for u, _ in members.values()}]
        rand = [int(x) for x in rng.choice(pool, size=N_RANDOM, replace=False)]
        cols = [s * down[:, u] for u, s in members.values()] + [down[:, u] for u in rand]
        writes = torch.stack(cols, dim=1)  # [hidden, m + N_RANDOM]
        J = lens["J"][layer].float() if layer != n_layers - 1 else None
        for lens_name, mat in (("j_lens", J), ("logit_lens", None)):
            vec = gain.unsqueeze(1) * (mat @ writes if mat is not None else writes)
            top, low = top_both(embed, vec)
            for j, name in enumerate(members):
                toks = decode(tok, top[:, j].tolist())
                readouts.setdefault(name, {"layer": layer, "unit": members[name][0], "sign": members[name][1]})
                readouts[name][lens_name] = {"top20": toks, "families": family_hits(toks)}
            m = len(members)
            hits = {fam: 0 for fam in FAMILIES}
            for j in range(m, m + N_RANDOM):  # each random neuron read with both signs: top of +w and top of -w
                for ids in (top[:, j], low[:, j]):
                    fh = family_hits(decode(tok, ids.tolist()))
                    for fam in FAMILIES:
                        hits[fam] += bool(fh[fam])
            nulls.setdefault(str(layer), {})[lens_name] = {fam: hits[fam] / (2 * N_RANDOM) for fam in FAMILIES}
        print(f"layer {layer} done {time.time() - t0:.0f}s", flush=True)
    verdicts = {}
    for key, (name, fam) in PREDICTED.items():
        r = readouts[name]
        null = nulls[str(r["layer"])]["j_lens"][fam]
        held = bool(r["j_lens"]["families"][fam])
        verdicts[key] = {"neuron": name, "family": fam, "status": "held" if held else "failed",
                         "null_rate": null, "specific": null < SPECIFIC}
    result = {"design": "PART111-DESIGN.md", "snapshot": str(SNAP), "lens": str(LENS),
              "lens_n_prompts": int(lens.get("n_prompts", -1)), "families": FAMILIES, "n_random": N_RANDOM,
              "readouts": readouts, "null_rates": nulls, "verdicts": verdicts, "elapsed_s": round(time.time() - t0, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jlens111.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: (v["status"], round(v["null_rate"], 3)) for k, v in verdicts.items()}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        sys.exit("usage: jlens111.py OUT_DIR")
    main(Path(sys.argv[1]))
