#!/usr/bin/env python3
"""Part 107: each stage's weight change by component group, the bf16 precision floor, and how much each stage changes
the later layers' reading of Part 92's unit push directions (PART107-DESIGN.md). Weights only, CPU.

Written by Claude (Opus 5.5), 2026-10-29, committed before any Part 107 run.
Usage: python part107/where107.py OUT.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part89.cos89 import CKPTS, Weights, weights_file  # noqa: E402  (registry commits and loader)

LAYERS = 16
STAGES = (("sft_change", "midtrain_51B", "sft"), ("dpo_change", "sft", "dpo"), ("rl_change", "dpo", "instruct"))
READERS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
LAYER_MATS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
RANDOM_PER_LAYER, SEED = 64, 107


def group_of(name: str) -> str:
    if name.startswith("model.embed_tokens"):
        return "embed"
    if name.startswith("lm_head"):
        return "unembed"
    if name == "model.norm.weight":
        return "final_norm"
    if "q_norm" in name or "k_norm" in name:
        return "attn_norm"
    if "self_attn." in name:
        return "attn"
    if "mlp." in name:
        return "mlp"
    if "layernorm" in name:
        return "layer_norm"
    return "other"


def bf16_steps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in bf16 representable steps between same-shape bf16 tensors (sign changes count as 2 or more)."""
    ai = a.contiguous().view(torch.int16).to(torch.int32)
    bi = b.contiguous().view(torch.int16).to(torch.int32)
    same_sign = (ai < 0) == (bi < 0)
    steps = (ai - bi).abs()
    return torch.where(same_sign, steps, torch.full_like(steps, 2))


def main(out_path: str):
    started = time.perf_counter()
    roots = {key: (root, prefix) for key, root, prefix in CKPTS}
    weights = {}
    for key in ("midtrain_51B", "sft", "dpo", "instruct"):
        shards = weights_file(*roots[key])
        if shards is None:
            raise SystemExit(f"checkpoint {key} is missing or incomplete")
        weights[key] = Weights(shards)
    names = sorted(n for n in weights_file(*roots["instruct"]) if n != "__bin__")

    neurons = json.loads((ROOT / "results/part92/step2/neurons.json").read_text(encoding="utf-8"))
    kinds = json.loads((ROOT / "results/part92/score92.json").read_text(encoding="utf-8"))["kinds"]
    units = [dict(id=r["id"], kind=kinds[r["id"]], layer=int(r["layer"]), neuron=int(r["neuron"]), persona=r["persona"])
             for r in neurons if kinds.get(r["id"]) in ("slot", "concept")]
    gen = torch.Generator().manual_seed(SEED)
    unit_layers = sorted({u["layer"] for u in units})
    randoms = {L: torch.randperm(8192, generator=gen)[:RANDOM_PER_LAYER].tolist() for L in unit_layers}
    # Directions: SFT down columns, normalised. Units first, then each layer's random neurons.
    dirs, owners = [], []
    for u in units:
        dirs.append((u["layer"], u["neuron"])); owners.append(("unit", u["id"]))
    for L in unit_layers:
        for n in randoms[L]:
            dirs.append((L, n)); owners.append(("random", L))
    down = {L: weights["sft"].get_tensor(f"model.layers.{L}.mlp.down_proj.weight").float() for L in unit_layers}
    U = torch.stack([down[L][:, n] / torch.linalg.vector_norm(down[L][:, n]) for L, n in dirs], dim=1)  # [2048, D]
    origin = torch.tensor([L for L, _n in dirs])

    out = {"stages": {}, "units": units, "random_per_layer": RANDOM_PER_LAYER, "seed": SEED}
    for stage, a_key, b_key in STAGES:
        groups, floor = {}, {"elements": 0, "changed": 0, "one_step": 0}
        num = torch.zeros(len(dirs)); den = torch.zeros(len(dirs))
        for name in names:
            try:
                a_raw, b_raw = weights[a_key].get_tensor(name), weights[b_key].get_tensor(name)
            except KeyError:
                continue
            if a_raw.shape != b_raw.shape:
                rows = min(a_raw.shape[0], b_raw.shape[0])
                if a_raw.shape[1:] != b_raw.shape[1:]:
                    continue
                a_raw, b_raw = a_raw[:rows], b_raw[:rows]
            a, b = a_raw.float(), b_raw.float()
            g = groups.setdefault(group_of(name), {"num": 0.0, "den": 0.0, "tensors": 0})
            g["num"] += float(torch.sum((b - a) ** 2)); g["den"] += float(torch.sum(a ** 2)); g["tensors"] += 1
            if any(name.endswith(f"{m}.weight") for m in LAYER_MATS) and a_raw.dtype == torch.bfloat16 == b_raw.dtype:
                steps = bf16_steps(a_raw, b_raw)
                floor["elements"] += steps.numel()
                floor["changed"] += int((steps > 0).sum()); floor["one_step"] += int((steps == 1).sum())
            for reader in READERS:
                for ell in range(LAYERS):
                    if name == f"model.layers.{ell}.{reader}.weight":
                        later = origin < ell
                        if bool(later.any()):
                            num[later] += torch.sum(((b - a) @ U[:, later]) ** 2, dim=0)
                            den[later] += torch.sum((a @ U[:, later]) ** 2, dim=0)
        # Unembedding through the final norm, reported separately.
        ga = weights[a_key].get_tensor("model.norm.weight").float()
        gb = weights[b_key].get_tensor("model.norm.weight").float()
        la = weights[a_key].get_tensor("lm_head.weight").float()
        lb = weights[b_key].get_tensor("lm_head.weight").float()
        rows = min(la.shape[0], lb.shape[0])
        ra, rb = la[:rows] * ga, lb[:rows] * gb
        r_lm = torch.sum(((rb - ra) @ U) ** 2, dim=0) / torch.sum((ra @ U) ** 2, dim=0)
        r = num / den
        out["stages"][stage] = {
            "from": a_key, "to": b_key,
            "groups": {k: {"rel_change": (v["num"] ** 0.5) / (v["den"] ** 0.5), "tensors": v["tensors"]}
                       for k, v in sorted(groups.items())},
            "precision_floor": {**floor, "changed_frac": floor["changed"] / max(floor["elements"], 1),
                                "one_step_frac_of_changed": floor["one_step"] / max(floor["changed"], 1)},
            "unit_R": {owners[i][1]: float(r[i]) for i in range(len(units))},
            "unit_R_unembed": {owners[i][1]: float(r_lm[i]) for i in range(len(units))},
            "random_R_by_layer": {str(L): [float(r[i]) for i in range(len(units), len(dirs)) if owners[i][1] == L]
                                  for L in unit_layers},
        }
        print(f"{stage}: done in {time.perf_counter() - started:.0f}s", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
