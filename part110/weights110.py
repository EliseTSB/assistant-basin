#!/usr/bin/env python3
"""Part 110 (PART110-DESIGN.md): how Qwen3.5-4B's maker neurons changed from Base to chat, from the weights alone.

Written by Claude (Opus 5.5), 2026-10-29. CPU only; safetensors are read straight from the local snapshots, and
nothing is generated. Usage: python weights110.py OUT_DIR. The test replaces the module globals (BASE_DIR,
CHAT_DIR, PART71 and the tables) on fixtures; the P110_* variables only set their defaults at import.
"""
import json, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open

HUB = Path("/mnt/e/neuron-probe/hf/hub")
BASE_DIR = Path(os.environ.get("P110_BASE", HUB / "models--Qwen--Qwen3.5-4B-Base/snapshots/1001bb4d826a52d1f399e183466143f4da7b741b"))
CHAT_DIR = Path(os.environ.get("P110_CHAT", HUB / "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"))
PART71 = Path(os.environ.get("P110_PART71", "results/part71/part71.json"))
PREFIX = "model.language_model."
MAKERS = {"L6 N753": (6, 753), "L7 N1169": (7, 1169), "L11 N8868": (11, 8868), "L29 N776": (29, 776)}
SECONDARY = {"L27 N3325": (27, 3325), "L24 N3949": (24, 3949), "L22 N2742": (22, 2742)}
GOOGLE_IDS = [4923, 14030]  # ' Google', 'Google' (Part 71's ids)
MATS = ("gate", "up", "down")
N_DRAWS, SEED, MATCH_K = 1000, 110110, 200
CHUNK = 16384


class Weights:
    """Tensor reader over one snapshot's safetensors shards."""

    def __init__(self, snap: Path):
        index = json.loads((snap / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        self.paths = {n: snap / f for n, f in index.items()}
        self.open = {}
        self.config = json.loads((snap / "config.json").read_text(encoding="utf-8"))

    def get(self, name: str) -> torch.Tensor:
        path = self.paths[name]
        if path not in self.open:
            self.open[path] = safe_open(str(path), framework="pt")
        return self.open[path].get_tensor(name)


def neuron_rows(w: Weights, layer: int) -> dict:
    """{mat: float64 [n_neurons, hidden]}, one row per neuron (down_proj transposed to its column per neuron).
    Float64 so that float32 rounding (about 1e-7 in a cosine) cannot pass for a small training change."""
    p = f"{PREFIX}layers.{layer}.mlp."
    return {"gate": w.get(p + "gate_proj.weight").double(), "up": w.get(p + "up_proj.weight").double(),
            "down": w.get(p + "down_proj.weight").double().T.contiguous()}


def layer_stats(base: dict, chat: dict) -> dict:
    """Per-neuron cosine, move (1 - cosine), chat/base norm ratio, and the base scale proxy used for matching."""
    out = {}
    for m in MATS:
        cos = torch.nn.functional.cosine_similarity(base[m], chat[m], dim=1).double().numpy()
        out[f"cos_{m}"] = cos
        out[f"move_{m}"] = 1.0 - cos
        out[f"ratio_{m}"] = (chat[m].norm(dim=1) / base[m].norm(dim=1)).double().numpy()
    out["move_mean"] = np.mean([out[f"move_{m}"] for m in MATS], axis=0)
    out["scale"] = sum(np.log(base[m].norm(dim=1).double().numpy()) for m in MATS)
    return out


def pct(values: np.ndarray, i: int) -> float:
    """Percent of the layer's neurons whose value is at most this neuron's (100 = moved the most)."""
    return float(100.0 * np.mean(values <= values[i]))


def set_test(stats: dict, members: list, matched: bool, rng) -> dict:
    """Mean of the members' move_mean against draws of one same-layer non-member per member (uniform or scale-matched)."""
    taken = set(members)
    pools = []
    for layer, unit in members:
        n = len(stats[layer]["move_mean"])
        others = np.array([u for u in range(n) if (layer, u) not in taken])
        if matched:
            gap = np.abs(stats[layer]["scale"][others] - stats[layer]["scale"][unit])
            others = others[np.argsort(gap, kind="stable")[:MATCH_K]]
        pools.append(others)
    observed = float(np.mean([stats[l]["move_mean"][u] for l, u in members]))
    draws = np.empty(N_DRAWS)
    for k in range(N_DRAWS):
        draws[k] = np.mean([stats[l]["move_mean"][int(rng.choice(pool))] for (l, _), pool in zip(members, pools)])
    return {"observed_mean_move": observed, "draw_mean": float(draws.mean()),
            "draw_q05_q95": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
            "p_more": (1 + int((draws >= observed).sum())) / (N_DRAWS + 1),
            "p_less": (1 + int((draws <= observed).sum())) / (N_DRAWS + 1),
            "pool_size": [int(len(p)) for p in pools]}


def readout(w: Weights, embed: torch.Tensor, layer: int, unit: int, sign: float) -> np.ndarray:
    """Part 71's readout from the weights: signed down column -> Qwen3.5 final RMSNorm (x * (1 + w)) -> tied unembedding."""
    v = sign * w.get(f"{PREFIX}layers.{layer}.mlp.down_proj.weight")[:, unit].float()
    eps = float(w.config.get("text_config", w.config).get("rms_norm_eps", 1e-6))
    v = v * torch.rsqrt(v.pow(2).mean() + eps) * (1.0 + w.get(PREFIX + "norm.weight").float())
    logits = torch.empty(embed.shape[0], dtype=torch.float64)
    for s in range(0, embed.shape[0], CHUNK):
        logits[s:s + CHUNK] = (embed[s:s + CHUNK].float() @ v).double()
    return logits.numpy()


def best_rank(logits: np.ndarray, ids: list) -> int:
    best = max(logits[i] for i in ids)
    return int(1 + np.sum(logits > best))


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(CHAT_DIR))


def verdicts(r: dict) -> dict:
    v = {}
    medians = [x for L in r["layers"].values() for x in L["median_cos"].values()]
    v["110.1"] = "held" if min(medians) > 0.9 else "failed"
    parent = v["110.1"] == "held"
    gate = all(x["chat_vs_part71_top20_shared"] >= 18 for x in r["readouts"].values())
    v["readout_gate"] = "passed" if gate else "failed"
    u, m = r["set_test"]["uniform"], r["set_test"]["matched"]
    v["110.2"] = ("held" if u["p_more"] > 0.05 else "failed") if parent else "NOT READ"
    v["110.3"] = ("held" if m["p_more"] > 0.05 and m["p_less"] > 0.05 else "failed") if parent else "NOT READ"
    readable = parent and gate
    g = r["readouts"].get("L27 N3325")
    v["110.4"] = ("held" if g["base"]["google_rank"] <= 10 else "failed") if readable and g else "NOT READ"
    v["110.5"] = ("held" if all(x["base_chat_top20_shared"] >= 10 for x in r["readouts"].values()) else "failed") \
        if readable else "NOT READ"
    return v


def main(out_dir: Path):
    t0 = time.time()
    named = {**MAKERS, **SECONDARY}
    layers = sorted({l for l, _ in named.values()})
    base_w, chat_w = Weights(BASE_DIR), Weights(CHAT_DIR)
    stats, layer_out = {}, {}
    self_min = None
    for layer in layers:
        b, c = neuron_rows(base_w, layer), neuron_rows(chat_w, layer)
        if self_min is None:  # implementation control: chat against itself must give cosine 1 everywhere
            self_min = float(min(layer_stats(c, c)[f"cos_{m}"].min() for m in MATS))
        st = layer_stats(b, c)
        stats[layer] = st
        layer_out[str(layer)] = {"median_cos": {m: float(np.median(st[f"cos_{m}"])) for m in MATS},
                                 "move_mean_quantiles": {q: float(np.quantile(st["move_mean"], float(q)))
                                                         for q in ("0.05", "0.5", "0.95", "0.99")},
                                 "median_ratio": {m: float(np.median(st[f"ratio_{m}"])) for m in MATS}}
        del b, c
        print(f"layer {layer} done {time.time() - t0:.0f}s", flush=True)
    neurons = {}
    for name, (l, u) in named.items():
        st = stats[l]
        n = len(st["move_mean"])
        near = np.argsort(np.abs(st["scale"] - st["scale"][u]), kind="stable")
        pool = np.array([x for x in near[:MATCH_K + 1] if x != u][:MATCH_K])
        neurons[name] = {"layer": l, "unit": u, "group": "maker" if name in MAKERS else "secondary",
                         "cos": {m: float(st[f"cos_{m}"][u]) for m in MATS},
                         "norm_ratio": {m: float(st[f"ratio_{m}"][u]) for m in MATS},
                         "move_mean": float(st["move_mean"][u]),
                         "pct_in_layer": {**{m: pct(st[f"move_{m}"], u) for m in MATS}, "mean": pct(st["move_mean"], u)},
                         "pct_vs_scale_matched": float(100.0 * np.mean(st["move_mean"][pool] <= st["move_mean"][u])),
                         "n_layer": n}
    rng = np.random.default_rng(SEED)
    makers = list(MAKERS.values())
    set_res = {"uniform": set_test(stats, makers, False, rng), "matched": set_test(stats, makers, True, rng)}
    p71 = json.loads(PART71.read_text(encoding="utf-8"))["named"]
    tok = load_tok()
    embeds = {"base": base_w.get(PREFIX + "embed_tokens.weight"), "chat": chat_w.get(PREFIX + "embed_tokens.weight")}
    readouts = {}
    for name, rec in p71.items():
        l, u, sign = int(rec["layer"]), int(rec["unit"]), float(rec["sign"])
        res = {"layer": l, "unit": u, "sign": sign}
        tops = {}
        for side, w in (("base", base_w), ("chat", chat_w)):
            logits = readout(w, embeds[side], l, u, sign)
            top = [int(i) for i in np.argsort(-logits, kind="stable")[:20]]
            tops[side] = top
            res[side] = {"google_rank": best_rank(logits, GOOGLE_IDS), "top20": [tok.decode([i]) for i in top]}
        res["base_chat_top20_shared"] = len(set(tops["base"]) & set(tops["chat"]))
        res["chat_vs_part71_top20_shared"] = sum((Counter(res["chat"]["top20"]) & Counter(rec["top20"])).values())
        readouts[name] = res
    unembed = {}
    cos_rows = torch.empty(embeds["base"].shape[0], dtype=torch.float64)
    for s in range(0, embeds["base"].shape[0], CHUNK):
        cos_rows[s:s + CHUNK] = torch.nn.functional.cosine_similarity(
            embeds["base"][s:s + CHUNK].float(), embeds["chat"][s:s + CHUNK].float(), dim=1).double()
    unembed["median_row_cos"] = float(cos_rows.median())
    unembed["google_row_cos"] = {str(i): float(cos_rows[i]) for i in GOOGLE_IDS}
    nb, nc = (1 + base_w.get(PREFIX + "norm.weight").float()), (1 + chat_w.get(PREFIX + "norm.weight").float())
    unembed["final_norm_scale_cos"] = float(torch.nn.functional.cosine_similarity(nb, nc, dim=0))
    result = {"design": "PART110-DESIGN.md", "base": str(BASE_DIR), "chat": str(CHAT_DIR), "layers": layer_out,
              "self_control_min_cos": self_min, "neurons": neurons, "set_test": set_res, "readouts": readouts,
              "unembed": unembed, "elapsed_s": round(time.time() - t0, 1)}
    result["verdicts"] = verdicts(result)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights110.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["verdicts"]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        sys.exit("usage: weights110.py OUT_DIR")
    main(Path(sys.argv[1]))
