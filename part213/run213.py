#!/usr/bin/env python3
"""Part 213 runner: OLMo 2 1B on CPU float32, the Part 92 slot/concept pushes, teacher-forced SpecLogit passes and
Stage 2 greedy free generations. Writes raw per-pass records; every gate and prediction is computed by score213.py from
those records only.

Built from PART213-DESIGN.md v5 (commit 7de53067) by the part213 builder, 2026-11-19. Not run against the model here.

Subcommands (each writes into --out):
  pilot   the 6 stratified units on `instruct` chat: the pilot's 1,056 forward passes (batch 16 vs batch 32) and 76
          greedy generations, obeying the pilot's output restriction (results/part213/pilot/pilot_summary.json only).
  stage1  every Stage 1 pass of every checkpoint (instruct, sft, dpo, FR), one JSONL per checkpoint, resumable.
  stage2  the 472 greedy generations at `instruct` (sealed concept rows written as `sealed`, never run).
  fr      builds FR = 2 * instruct - dpo in-place in float32 and runs its Stage 1 rows only.

Run as `python -I -B part213/run213.py <sub> --out <dir>` from the repo root (the repo root is inserted here, as
part213/build_targets213.py does).

Design ambiguities met here are listed in part213/BUILD-REPORT.md with line numbers; the most literal reading is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TARGETS = ROOT / "part213" / "targets213.json"
TARGETS_SHA256 = "7edb3c6253470d157896931083c239253930be08f16a5b9348d6c67d3b431892"
PUSHES_PT = ROOT / "results" / "part99-pushes.pt"
PUSHES_JSON = ROOT / "results" / "part99-pushes.json"
PART106_INSTRUCT = ROOT / "results" / "part106" / "lp-instruct.jsonl"
HUB = Path("/mnt/e/neuron-probe/hf/hub")

CHECKPOINTS = ("sft", "dpo", "instruct")  # FR is built from dpo + instruct in place
REPOS = {
    "sft": "models--allenai--OLMo-2-0425-1B-SFT",
    "dpo": "models--allenai--OLMo-2-0425-1B-DPO",
    "instruct": "models--allenai--OLMo-2-0425-1B-Instruct",
}
OPEN_ALPHA = 0.8          # Part 99's stored vectors are alpha-0.8; the hook scales by alpha / 0.8 (design l.117)
FD_EPS = 0.05             # design l.36, l.225
PILOT_EPS = 0.025         # design l.229
PILOT_UNITS = ("p92-05", "p92-04", "p92-37", "p92-00", "p92-08", "p92-39")  # design l.377-379
GATE2_ANCHORS = ("p92-02", "p92-05", "p92-00", "p92-08")                    # design l.341-342
SELF_HELD = ("self_h0", "self_h1", "self_h2", "self_h3")
THIRD_SLOT = ("third_0", "third_1", "third_2", "third_3")
OFFDIAG_PROMPTS = ("self_h0", "third_0", "third_2")                          # design l.218, l.239
SEALED_CELLS = ("self_select", "free_self_0", "free_cold_0")                # design l.66-71
MAX_NEW_TOKENS = 40

# What each pass records. The Stage 1 core pass records everything the metrics need; the pilot's restricted passes
# record only what the pilot's output restriction allows (design l.384-388).
CORE_FIELDS = ("first_logits", "lp_sum", "lp_mean", "lp_first", "entropy", "top1", "top1_id")
FD_FIELDS = ("first_logits",)


# ---------------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_targets() -> dict:
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))
    digest = sha256_file(TARGETS)
    if digest != TARGETS_SHA256:
        raise SystemExit(f"part213/targets213.json is {digest}, not the registered {TARGETS_SHA256}; load, never rebuild")
    return targets


def snapshot_dir(ckpt: str) -> Path:
    """The one local HF snapshot of a checkpoint; its name is the full commit (checked against part85 later)."""
    from part85 import checkpoints as registry

    prefix = registry.BY_KEY[ckpt].commit_prefix
    hits = [d for d in (HUB / REPOS[ckpt] / "snapshots").iterdir() if d.name.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"{ckpt}: expected one snapshot starting {prefix}, found {len(hits)}")
    return hits[0]


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)  # ru_maxrss is KiB on Linux


def load_model_fp32(snapshot: Path):
    """OLMo 2 1B in torch.float32 on CPU with eager attention (design l.102-103); part92's loader is bf16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), torch_dtype=torch.float32, local_files_only=True, attn_implementation="eager",
    ).to("cpu")
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def load_push_vectors(units: list[dict], model, checkpoint: str):
    """Frozen alpha-0.8 neuron and paired random_0 vectors; random_1..3 only at instruct (design l.114-124)."""
    import torch
    from part92 import common92

    pushes = torch.load(PUSHES_PT, map_location="cpu")
    norms = json.loads(PUSHES_JSON.read_text(encoding="utf-8"))["neuron_norms"]
    layers = common92.get_layers(model)
    vectors = {}
    for index, unit in enumerate(units):
        uid = unit["id"]
        weight = layers[int(unit["layer"])].mlp.down_proj.weight
        vectors[(uid, "neuron")] = pushes[f"{uid}|neuron"].float()
        vectors[(uid, "random_0")] = pushes[f"{uid}|random"].float()
        if checkpoint == "instruct":
            for m in (1, 2, 3):
                seed = 21310 + 100 * m + index  # design l.120: unit_index in targets213.json catalog order
                direction = common92.random_directions(weight, 1, seed, float(norms[uid]))[0]
                vectors[(uid, f"random_{m}")] = direction.float()
    return vectors


def scale_vector(vector, alpha: float):
    return vector * (float(alpha) / OPEN_ALPHA)


def torch_stack(vectors):
    import torch

    return torch.stack([v.float() for v in vectors])


# ---------------------------------------------------------------------------------------------------------------------
# forward passes
# ---------------------------------------------------------------------------------------------------------------------

def prompt_ids(tokenizer, question: str) -> list[int]:
    from part92 import common92

    return common92.prompt_ids(tokenizer, question, "chat")


def forward_batch(model, tokenizer, rows: list[dict], layer=None, vectors=None, np_specs=None):
    """One right-padded forward pass over rows; returns per-row logits at the unit's first-content position.

    rows: [{"ids": [...], "first_pos": int, "content_pos": [int...], "target_ids": [...]}]. Right padding with
    attention_mask 0 (design l.130-132); the read index is prefix_len + len(lead_ids) - 1 (design l.132).
    np_specs: optional {unit_id: (content_pos, target_ids)}; every row then also scores every listed unit's full NP
    from the same pass (the design's zero-extra-pass offdiag_8 baseline, l.204-207), returned in `lp_by_unit`.
    """
    import torch

    pad = int(tokenizer.pad_token_id)
    width = max(len(r["ids"]) for r in rows)
    ids = torch.tensor([r["ids"] + [pad] * (width - len(r["ids"])) for r in rows], dtype=torch.long)
    mask = torch.tensor([[1] * len(r["ids"]) + [0] * (width - len(r["ids"])) for r in rows], dtype=torch.long)
    handle = None
    if vectors is not None:
        if vectors.shape[0] != len(rows):
            raise ValueError("one row vector per row is required")
        handle = _add_row_hook(layer, vectors)
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    finally:
        if handle is not None:
            handle.remove()
    np_logp = {}
    if np_specs:
        for uid, (positions, target_ids) in np_specs.items():
            if not positions:
                continue
            np_logp[uid] = torch.log_softmax(logits[:, torch.tensor(positions, dtype=torch.long)], dim=-1)
    out = []
    for row_no, row in enumerate(rows):
        seq = logits[row_no]
        first = seq[int(row["first_pos"])]
        n = len(row["target_ids"])
        if n:
            positions = torch.tensor([int(p) for p in row["content_pos"]], dtype=torch.long)
            logp = torch.log_softmax(seq[positions], dim=-1)
            values = logp[torch.arange(len(positions)), torch.tensor(row["target_ids"], dtype=torch.long)]
            lp_sum, lp_first = float(values.sum()), float(values[0])
            lp_mean = lp_sum / n
        else:
            lp_sum = lp_mean = lp_first = 0.0
        probs = torch.softmax(first, dim=-1)
        entry = {
            "first_logits": first.tolist(),
            "lp_sum": lp_sum, "lp_mean": lp_mean, "lp_first": lp_first,
            "entropy": float(-(probs * torch.log(probs.clamp_min(1e-30))).sum()),
            "top1": float(probs.max()),
            "top1_id": int(probs.argmax()),
        }
        if np_specs:
            by_unit = {}
            for uid, (positions, target_ids) in np_specs.items():
                if not positions:
                    continue
                values = np_logp[uid][row_no, torch.arange(len(positions)), torch.tensor(target_ids, dtype=torch.long)]
                total = float(values.sum())
                by_unit[uid] = {"lp_sum": total, "lp_mean": total / len(positions)}
            entry["lp_by_unit"] = by_unit
        out.append(entry)
    return out


def _add_row_hook(layer, vectors):
    """part92/common92.add_row_hook, kept here only so the runner does not import part92's bf16 loader path."""
    from part92 import common92

    return common92.add_row_hook(layer, vectors)


def make_row(tokenizer, question: str, opening: str, unit: dict) -> dict:
    """chat_prefix + opening + " " + np_text, ids asserted to be the concatenation (design l.125-132, l.344-346)."""
    prefix = prompt_ids(tokenizer, question)
    open_ids = tokenizer.encode(opening, add_special_tokens=False)
    target_ids = [int(t) for t in unit["target_ids"]]
    lead = [int(t) for t in unit["lead_ids"]]
    ids = prefix + open_ids + target_ids
    if ids[len(prefix): len(prefix) + len(open_ids)] != open_ids:
        raise RuntimeError(f"chat/opening merge on {question!r}")
    if ids[len(prefix) + len(open_ids):] != target_ids:
        raise RuntimeError(f"chat/target merge on {unit['id']} / {question!r}")
    first_pos = len(prefix) + len(lead) - 1
    content_start = int(unit["content_start"])
    content_pos = [len(prefix) + content_start - 1 + i for i in range(len(target_ids) - content_start)]
    return {"ids": ids, "first_pos": first_pos, "content_pos": content_pos,
            "target_ids": target_ids[content_start:], "prefix_len": len(prefix), "opening_ids": open_ids}


def np_specs_for(tokenizer, question: str, opening: str, units, cache: dict | None = None) -> dict:
    """{unit_id: (content_pos, target_ids)} for every unit's full NP after one prompt's opening (design l.204-207).

    `cache` keyed by (question, opening) keeps the per-prompt tokenisation out of the per-batch loop.
    """
    key = (question, opening)
    if cache is not None and key in cache:
        return cache[key]
    specs = {}
    for unit in units:
        row = make_row(tokenizer, question, opening, unit)
        specs[unit["id"]] = (row["content_pos"], row["target_ids"])
    if cache is not None:
        cache[key] = specs
    return specs


def keep_fields(result: dict, fields) -> dict:
    """Keep only the recorded fields; a field outside `fields` is written as null, never silently filled."""
    return {k: (result[k] if k in fields else None) for k in CORE_FIELDS}


# ---------------------------------------------------------------------------------------------------------------------
# pass plans (the design's counted grids)
# ---------------------------------------------------------------------------------------------------------------------

def stage1_plan(targets: dict, checkpoint: str) -> list[tuple[str, str, str, float, str]]:
    """(prompt_id, arm, dose, alpha, fields) for every Stage 1 pass of one checkpoint.

    instruct (design l.232-241): none + neuron +0.8 + random_0..3 +0.8 on 17 prompts; neuron -0.8 and random_0 -0.8 on
    self_held; neuron/random_0 ±0.05 on self_held + third_slot; the offdiag_8 full-NP passes are the neuron +0.8 core
    passes on self_h0/third_0/third_2 (the core pass already scores all 39 units' full NPs, design l.204-207).
    sft/dpo/FR (design l.242-244): none, neuron +0.8, random_0 +0.8 on self_held + third_slot.
    """
    prompts = {p["id"]: p for cell, plist in targets["stage1_prompts"].items() for p in plist}
    plan = []
    if checkpoint == "instruct":
        for pid in prompts:
            plan.append((pid, "none", "0.0", 0.0, CORE_FIELDS))
            plan.append((pid, "neuron", "+0.8", OPEN_ALPHA, CORE_FIELDS))
            for m in range(4):
                plan.append((pid, f"random_{m}", "+0.8", OPEN_ALPHA, CORE_FIELDS))
        for pid in SELF_HELD:
            plan.append((pid, "neuron", "-0.8", -OPEN_ALPHA, CORE_FIELDS))
            plan.append((pid, "random_0", "-0.8", -OPEN_ALPHA, CORE_FIELDS))
        for pid in SELF_HELD + THIRD_SLOT:
            for arm in ("neuron", "random_0"):
                plan.append((pid, arm, "+0.05", FD_EPS, FD_FIELDS))
                plan.append((pid, arm, "-0.05", -FD_EPS, FD_FIELDS))
    else:
        for pid in SELF_HELD + THIRD_SLOT:
            plan.append((pid, "none", "0.0", 0.0, CORE_FIELDS))
            plan.append((pid, "neuron", "+0.8", OPEN_ALPHA, CORE_FIELDS))
            plan.append((pid, "random_0", "+0.8", OPEN_ALPHA, CORE_FIELDS))
    return plan


def is_sealed(unit: dict, prompt_id: str) -> bool:
    """Part 180 run-order seal (design l.66-71): concept-unit rows of self_select and free_self_0/free_cold_0.

    The design names the cells `self_select` and `free_self_0`/`free_cold_0`; the prompt ids are
    `self_select_0`, `free_self_0`, `free_cold_0`, so the Stage 1 cell and the two Stage 2 prompts are both covered.
    """
    return unit["kind_haiku"] == "concept" and (prompt_id in SEALED_CELLS or prompt_id == "self_select_0")


# ---------------------------------------------------------------------------------------------------------------------
# record IO
# ---------------------------------------------------------------------------------------------------------------------

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record(unit: dict, prompt_id: str, arm: str, dose: str, alpha: float, result: dict, extra: dict | None = None):
    row = {"unit": unit["id"], "kind": unit["kind_haiku"], "layer": int(unit["layer"]), "neuron": int(unit["neuron"]),
           "prompt_id": prompt_id, "arm": arm, "dose": dose, "alpha": float(alpha), "status": "ok"}
    row.update({k: result.get(k) for k in CORE_FIELDS})
    if result.get("lp_by_unit"):
        row["lp_by_unit"] = result["lp_by_unit"]  # every unit's full-NP score from this pass (offdiag_8 baselines)
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------------------------------------------------
# run.json
# ---------------------------------------------------------------------------------------------------------------------

def provenance(checkpoint: str, snapshot: Path) -> dict:
    from part85 import checkpoints as registry

    resolved = registry.resolve_checkpoint(checkpoint)
    registry.validate_snapshot_commit(snapshot, resolved["commit"])
    return resolved


def tokenizer_digests() -> dict:
    """Gate 1: identical tokenizer.json / tokenizer_config.json across sft, dpo, instruct (design l.128-129)."""
    out = {}
    for ckpt in CHECKPOINTS:
        snap = snapshot_dir(ckpt)
        out[ckpt] = {name: (sha256_file(snap / name) if (snap / name).is_file() else None)
                     for name in ("tokenizer.json", "tokenizer_config.json")}
    return out


def residual_norm_ratio(model, tokenizer) -> float:
    """Each checkpoint's residual median norm relative to instruct (design l.123-124)."""
    from part92 import common92

    return float(common92.residual_medians(model, tokenizer, "chat")[0])


def write_run_json(out: Path, checkpoint: str, snapshot: Path, model, tokenizer, extra: dict) -> dict:
    manifest = {
        "part": 213,
        "checkpoint": checkpoint,
        "design": "PART213-DESIGN.md v5 (commit 7de53067)",
        "snapshot": str(snapshot),
        "snapshot_commit": snapshot.name,
        "resolved": provenance(checkpoint, snapshot),
        "dtype": "float32",
        "attn_implementation": "eager",
        "device": "cpu",
        "tool_sha256": {"part213/run213.py": sha256_file(Path(__file__).resolve())},
        "inputs_sha256": {
            "part213/targets213.json": sha256_file(TARGETS),
            "results/part99-pushes.pt": sha256_file(PUSHES_PT),
            "results/part99-pushes.json": sha256_file(PUSHES_JSON),
            "results/part106/lp-instruct.jsonl": sha256_file(PART106_INSTRUCT),
        },
        "tokenizer_sha256": tokenizer_digests(),
        "residual_median_norm_layer0": residual_norm_ratio(model, tokenizer),
        "residual_median_norm_reference_note": "instruct is the reference for the ratio; score213 compares across run.json",
        "peak_rss_gb": round(peak_rss_gb(), 3),
        "sealed_cells": list(SEALED_CELLS),
        "seal_rule": "concept-unit rows of self_select and free_self_0/free_cold_0 are written as status=sealed, never run",
        "gate2_anchors": list(GATE2_ANCHORS),
        **extra,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------------------------------------------------
# stage 1
# ---------------------------------------------------------------------------------------------------------------------

def run_stage1_pass(model, tokenizer, units, vectors, plan, prompts, batch, layer_of, out_path, checkpoint):
    """Execute one checkpoint's Stage 1 plan, writing one JSONL row per (unit, prompt, arm)."""
    started = time.perf_counter()
    rows = []
    spec_cache = {}
    for pid, arm, dose, alpha, fields in plan:
        prompt = prompts[pid]
        work = []
        for unit in units:
            if is_sealed(unit, pid):
                rows.append({"unit": unit["id"], "kind": unit["kind_haiku"], "prompt_id": pid, "arm": arm,
                             "dose": dose, "alpha": alpha, "status": "sealed"})
                continue
            work.append(unit)
        for begin in range(0, len(work), batch):
            chunk = work[begin: begin + batch]
            layer = layer_of(chunk[0])
            if any(layer_of(u) is not layer for u in chunk):
                raise RuntimeError("a batch must share one decoder layer")
            batch_rows = [make_row(tokenizer, prompt["question"], prompt["opening"], u) for u in chunk]
            # Right-padded rows of one decoder layer in one forward pass (design l.130-132, l.406-407); a `none` arm
            # carries no hook, so it is one pass per prompt and layer.
            specs = (np_specs_for(tokenizer, prompt["question"], prompt["opening"], units, spec_cache)
                     if fields is CORE_FIELDS else None)
            if arm == "none":
                raw = forward_batch(model, tokenizer, batch_rows, np_specs=specs)
            else:
                vectors = torch_stack([scale_vector(vectors[(u["id"], arm)], alpha) for u in chunk])
                raw = forward_batch(model, tokenizer, batch_rows, layer=layer, vectors=vectors, np_specs=specs)
            for unit, result in zip(chunk, raw):
                rows.append(record(unit, pid, arm, dose, alpha, keep_fields(result, fields),
                                   {"n_content_tokens": int(unit["n_content_tokens"]),
                                    "n_target_tokens": int(unit["n_target_tokens"])}))
        print(f"{checkpoint}: {pid} {arm} {dose} done, {len(rows)} rows, {time.perf_counter() - started:.0f}s", flush=True)
        write_jsonl(out_path, rows)  # resume granularity: one rewritten file per (prompt, arm)
    return rows


def cmd_stage1(args) -> int:
    targets = load_targets()
    units = targets["units"]
    prompts = {p["id"]: p for plist in targets["stage1_prompts"].values() for p in plist}
    out = Path(args.out)
    done = []
    for checkpoint in CHECKPOINTS:
        out_path = out / f"stage1-{checkpoint}.jsonl"
        if out_path.exists() and not args.force:
            print(f"{checkpoint}: {out_path} exists, skipping (resume rule, design l.413-414)", flush=True)
            done.append(checkpoint)
            continue
        snap = snapshot_dir(checkpoint)
        tokenizer, model = load_model_fp32(snap)
        from part92 import common92

        layers = common92.get_layers(model)
        vectors = load_push_vectors(units, model, checkpoint)
        plan = stage1_plan(targets, checkpoint)
        run_stage1_pass(model, tokenizer, units, vectors, plan, prompts, args.batch,
                        lambda u: layers[int(u["layer"])], out_path, checkpoint)
        write_run_json(out, checkpoint, snap, model, tokenizer, {
            "stage1_plan_passes": len(plan) * len(units), "batch": args.batch})
        done.append(checkpoint)
        del model
    if args.with_fr:
        return cmd_fr(argparse.Namespace(out=args.out, batch=args.batch, force=args.force))
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# FR: 2 * instruct - dpo, in place, float32
# ---------------------------------------------------------------------------------------------------------------------

def build_fr(instruct_snapshot: Path, dpo_snapshot: Path):
    """Load instruct (~5.9 GB) and stream dpo's safetensors one tensor at a time: param.mul_(2).sub_(dpo) (l.108-111)."""
    import torch
    from safetensors import safe_open

    tokenizer, model = load_model_fp32(instruct_snapshot)
    state = model.state_dict()
    files = sorted(dpo_snapshot.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"no safetensors in {dpo_snapshot}")
    seen = set()
    with torch.no_grad():
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key not in state:
                        continue
                    tensor = handle.get_tensor(key)
                    target = state[key]
                    if tuple(tensor.shape) != tuple(target.shape):
                        raise RuntimeError(f"{key}: dpo shape {tuple(tensor.shape)} != instruct {tuple(target.shape)}")
                    target.mul_(2.0).sub_(tensor.float())
                    seen.add(key)
    missing = sorted(set(state) - seen)
    if missing:
        raise SystemExit(f"FR could not be built: {len(missing)} instruct tensors absent from dpo, e.g. {missing[:3]}")
    return tokenizer, model


def cmd_fr(args) -> int:
    targets = load_targets()
    units = targets["units"]
    prompts = {p["id"]: p for plist in targets["stage1_prompts"].values() for p in plist}
    out = Path(args.out)
    out_path = out / "stage1-fr.jsonl"
    if out_path.exists() and not getattr(args, "force", False):
        print(f"FR: {out_path} exists, skipping", flush=True)
        return 0
    tokenizer, model = build_fr(snapshot_dir("instruct"), snapshot_dir("dpo"))
    from part92 import common92

    layers = common92.get_layers(model)
    vectors = load_push_vectors(units, model, "fr")  # random_0 only outside instruct
    plan = stage1_plan(targets, "fr")
    run_stage1_pass(model, tokenizer, units, vectors, plan, prompts, args.batch,
                    lambda u: layers[int(u["layer"])], out_path, "fr")
    write_run_json(out, "fr", snapshot_dir("instruct"), model, tokenizer, {
        "fr_formula": "2 * instruct - dpo", "fr_built_in_place": True, "fr_dtype": "float32",
        "fr_sources": {"instruct": snapshot_dir("instruct").name, "dpo": snapshot_dir("dpo").name},
        "stage1_plan_passes": len(plan) * len(units), "batch": args.batch})
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# stage 2: greedy free generations
# ---------------------------------------------------------------------------------------------------------------------

def stage2_plan(targets: dict) -> list[tuple[str, str, str, float]]:
    """39 units x (neuron +0.8, neuron -0.8, random_0 +0.8) x 4 prompts, plus 4 unpushed baselines (l.250-253)."""
    plan = [(p["id"], "none", "0.0", 0.0) for p in targets["stage2_free_prompts"]]
    for unit in targets["units"]:
        for p in targets["stage2_free_prompts"]:
            for arm, dose, alpha in (("neuron", "+0.8", OPEN_ALPHA), ("neuron", "-0.8", -OPEN_ALPHA),
                                     ("random_0", "+0.8", OPEN_ALPHA)):
                plan.append((p["id"], arm, dose, alpha))
    return plan


def generate_one(model, tokenizer, layer, question: str, vector) -> str:
    """Greedy 40-token generation in the chat frame with no forced opening (design l.253-254)."""
    import torch

    ids = prompt_ids(tokenizer, question)
    handle = _add_row_hook(layer, vector.unsqueeze(0)) if vector is not None else None
    try:
        with torch.inference_mode():
            gen = model.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                                 do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                 pad_token_id=int(tokenizer.pad_token_id))
    finally:
        if handle is not None:
            handle.remove()
    return tokenizer.decode(gen[0, len(ids):], skip_special_tokens=True)


def cmd_stage2(args) -> int:
    targets = load_targets()
    units = {u["id"]: u for u in targets["units"]}
    prompts = {p["id"]: p for p in targets["stage2_free_prompts"]}
    out = Path(args.out)
    out_path = out / "stage2-instruct.jsonl"
    if out_path.exists() and not args.force:
        print(f"stage2: {out_path} exists, skipping", flush=True)
        return 0
    snap = snapshot_dir("instruct")
    tokenizer, model = load_model_fp32(snap)
    from part92 import common92

    layers = common92.get_layers(model)
    vectors = load_push_vectors(list(units.values()), model, "instruct")
    rows = []
    for pid, arm, dose, alpha in stage2_plan(targets):
        prompt = prompts[pid]
        if arm == "none":
            text = generate_one(model, tokenizer, None, prompt["question"], None)
            rows.append({"unit": None, "prompt_id": pid, "context": prompt["context"], "arm": "none", "dose": dose,
                         "alpha": 0.0, "status": "ok", "answer": text})
        else:
            for unit in units.values():
                if is_sealed(unit, pid):
                    rows.append({"unit": unit["id"], "prompt_id": pid, "context": prompt["context"], "arm": arm,
                                 "dose": dose, "alpha": alpha, "status": "sealed"})
                    continue
                vector = scale_vector(vectors[(unit["id"], arm)], alpha)
                text = generate_one(model, tokenizer, layers[int(unit["layer"])], prompt["question"], vector)
                rows.append({"unit": unit["id"], "kind": unit["kind_haiku"], "prompt_id": pid,
                             "context": prompt["context"], "arm": arm, "dose": dose, "alpha": alpha,
                             "status": "ok", "answer": text})
        print(f"stage2: {pid} {arm} {dose} done, {len(rows)} rows", flush=True)
        write_jsonl(out_path, rows)
    write_run_json(out, "instruct", snap, model, tokenizer, {"stage": "stage2", "max_new_tokens": MAX_NEW_TOKENS})
    return 0


# ---------------------------------------------------------------------------------------------------------------------
# pilot: the 6 stratified units, the restricted summary only
# ---------------------------------------------------------------------------------------------------------------------

def pilot_plan() -> list[tuple[str, str, str, float, str]]:
    """612 core + 48 negative + 192 ±0.05 + 192 ±0.025 + 12 Gate 2 anchor passes = 1,056 (design l.379-382)."""
    plan = []
    for pid in ("self_select_0", "self_h0", "self_h1", "self_h2", "self_h3", "third_0", "third_1", "third_2",
                "third_3", "mention_0", "mention_1", "mention_2", "mention_3", "copula_nonid_0", "copula_nonid_1",
                "self_nonid_0", "self_nonid_1"):
        plan.append((pid, "none", "0.0", 0.0, CORE_FIELDS))
        plan.append((pid, "neuron", "+0.8", OPEN_ALPHA, CORE_FIELDS))
        for m in range(4):
            plan.append((pid, f"random_{m}", "+0.8", OPEN_ALPHA, CORE_FIELDS))
    for pid in SELF_HELD:
        plan.append((pid, "neuron", "-0.8", -OPEN_ALPHA, CORE_FIELDS))
        plan.append((pid, "random_0", "-0.8", -OPEN_ALPHA, CORE_FIELDS))
    for pid in SELF_HELD + THIRD_SLOT:
        for arm in ("neuron", "random_0"):
            for eps in (PILOT_EPS, FD_EPS):
                plan.append((pid, arm, f"+{eps}", eps, FD_FIELDS))
                plan.append((pid, arm, f"-{eps}", -eps, FD_FIELDS))
    return plan


def cmd_pilot(args) -> int:
    """The pilot writes ONLY the restricted summary (design l.384-388): no L_z, tau, or pushed persona texts."""
    targets = load_targets()
    units = {u["id"]: u for u in targets["units"]}
    pilot_units = [units[uid] for uid in PILOT_UNITS]
    prompts = {p["id"]: p for plist in targets["stage1_prompts"].values() for p in plist}
    out = Path(args.out)
    snap = snapshot_dir("instruct")
    tokenizer, model = load_model_fp32(snap)
    from part92 import common92

    layers = common92.get_layers(model)
    vectors = load_push_vectors(list(units.values()), model, "instruct")
    summary: dict = {"stage": "pilot", "snapshot_commit": snap.name, "dtype": "float32", "units": list(PILOT_UNITS),
                     "sealed_skipped": [], "gate1": {}, "gate2": {}, "throughput": {}, "step_size": {},
                     "stage2_baseline_answers": [], "stage2_broken": {}}
    started = time.perf_counter()

    # Gate 1: zero-vector hook at identical batch shape; batch-padded vs batch-1 unpadded; finite values.
    pid = "self_h0"
    layer = layers[int(pilot_units[0]["layer"])]
    rows = [make_row(tokenizer, prompts[pid]["question"], prompts[pid]["opening"], u) for u in pilot_units
            if int(u["layer"]) == int(pilot_units[0]["layer"])]
    if rows:
        plain = forward_batch(model, tokenizer, rows)
        import torch

        zero = forward_batch(model, tokenizer, rows, layer=layer, vectors=torch.zeros(len(rows), model.config.hidden_size))
        diff_hook = max(abs(a - b) for r0, r1 in zip(plain, zero)
                        for a, b in zip(r0["first_logits"], r1["first_logits"]))
        one = [forward_batch(model, tokenizer, [r])[0] for r in rows]
        diff_batch = max(abs(a - b) for r0, r1 in zip(plain, one)
                         for a, b in zip(r0["first_logits"], r1["first_logits"]))
        finite = all(math.isfinite(v) for r in plain for v in r["first_logits"])
        summary["gate1"] = {"zero_hook_max_abs_diff": diff_hook, "batch_padded_vs_batch1_max_abs_diff": diff_batch,
                            "finite": finite,
                            "passed": diff_hook == 0.0 and diff_batch < 1e-4 and finite}
        summary["pilot_gate1"] = summary["gate1"]  # score213's Gate 1 reads this key from run.json

    # Gate 2: the 4 Part 106 anchors, chat_prefix + anchor string, ids asserted, lp_sum within 0.50 nats.
    stored = {}
    for line in PART106_INSTRUCT.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        stored[(row["id"], row["unit"])] = row
    gate2 = []
    for uid in GATE2_ANCHORS:
        unit = units[uid]
        anchor = next(a["target"] for a in targets["gate2_anchors"] if a["id"] == uid)
        prefix = prompt_ids(tokenizer, "Who are you?")
        anchor_ids = tokenizer.encode(anchor, add_special_tokens=False)
        full = prefix + anchor_ids
        if tokenizer(full, add_special_tokens=False)["input_ids"] != full:
            raise RuntimeError(f"Gate 2 anchor ids do not equal chat_ids + encode(anchor) for {uid}")
        target_ids = anchor_ids
        row_spec = {"ids": full, "first_pos": len(full) - 1,
                    "content_pos": [len(prefix) - 1 + i for i in range(len(target_ids))], "target_ids": target_ids}
        entry = {"id": uid, "anchor": anchor, "n_tokens": len(target_ids)}
        for arm in ("none", "neuron", "random_0"):
            vector = None if arm == "none" else scale_vector(vectors[(uid, arm)], OPEN_ALPHA)
            result = forward_batch(model, tokenizer, [row_spec], layer=layers[int(unit["layer"])], vectors=vector)[0]
            key = "random" if arm == "random_0" else arm
            entry[f"lp_sum_{arm}"] = result["lp_sum"]
            entry[f"part106_lp_sum_{arm}"] = stored[(uid, key)]["lp_sum"]
            entry[f"abs_diff_{arm}"] = abs(result["lp_sum"] - stored[(uid, key)]["lp_sum"])
        entry["ordering_ok"] = (entry["lp_sum_neuron"] > entry["lp_sum_random_0"] > entry["lp_sum_none"])
        entry["passed"] = (entry["ordering_ok"] and all(entry[f"abs_diff_{a}"] < 0.50 for a in ("none", "neuron", "random_0")))
        gate2.append(entry)
    summary["gate2"] = {"anchors": gate2, "passed": all(e["passed"] for e in gate2)}
    summary["gate2_anchors_detail"] = gate2  # score213's Gate 2 reads these (score213.gate2)
    summary["tokenizer_sha256"] = tokenizer_digests()  # Gate 1's identical-digest check (design l.128-129)

    # Throughput and step-size stability: batch 16 vs batch 32 on the core plan, and ±0.05 vs ±0.025 on J_N.
    j_eps = {PILOT_EPS: {}, FD_EPS: {}}
    for batch in (16, 32):
        t0 = time.perf_counter()
        n = 0
        for pid2, arm, dose, alpha, fields in pilot_plan():
            prompt = prompts[pid2]
            work = [u for u in pilot_units if not is_sealed(u, pid2)]
            if len(work) < len(pilot_units):
                summary["sealed_skipped"].append(f"{pid2}|{arm}|{dose}")
            for begin in range(0, len(work), batch):
                chunk = work[begin: begin + batch]
                layer = layers[int(chunk[0]["layer"])]
                if any(int(u["layer"]) != int(chunk[0]["layer"]) for u in chunk):
                    raise RuntimeError("a batch must share one decoder layer")
                batch_rows = [make_row(tokenizer, prompt["question"], prompt["opening"], u) for u in chunk]
                if arm == "none":
                    raw = forward_batch(model, tokenizer, batch_rows)
                else:
                    raw = forward_batch(model, tokenizer, batch_rows, layer=layer,
                                        vectors=torch_stack([scale_vector(vectors[(u["id"], arm)], alpha) for u in chunk]))
                n += len(chunk)
                if arm == "neuron" and fields is FD_FIELDS:
                    for unit, result in zip(chunk, raw):
                        j_eps[float(dose)].setdefault((unit["id"], pid2), {})[float(alpha)] = result["first_logits"]
        elapsed = time.perf_counter() - t0
        summary["throughput"][f"batch_{batch}"] = {"passes": n, "seconds": round(elapsed, 2),
                                                   "passes_per_s": round(n / elapsed, 3)}
    # The pilot's stopping rule reads this correlation (design l.388-390); only the correlation is kept, never J_N.
    pairs = [(uid, pid2) for (uid, pid2) in j_eps[FD_EPS] if uid in j_eps[PILOT_EPS] and pid2 in j_eps[PILOT_EPS][uid]]
    if pairs:
        j05, j025 = [], []
        for uid, pid2 in pairs:
            unit = units[uid]
            own = int(unit["first_content_token_id"])
            for eps, store in ((FD_EPS, j05), (PILOT_EPS, j025)):
                plus, minus = j_eps[eps][(uid, pid2)][eps], j_eps[eps][(uid, pid2)][-eps]
                store.append((plus[own] - minus[own]) / (2 * eps))
        summary["step_size"] = {"n_pairs": len(pairs), "pearson_r_J_N_0.05_vs_0.025": round(pearson(j05, j025), 4),
                                "spearman_r_J_N_0.05_vs_0.025": round(spearman(j05, j025), 4),
                                "passed": spearman(j05, j025) >= 0.90}

    # Stage 2: the 4 unpushed baseline answers and mechanical broken-answer counts (design l.386-387). The seal covers
    # concept-unit pushed rows only (design l.66-71), so the unpushed baselines are run.
    for p in targets["stage2_free_prompts"]:
        text = generate_one(model, tokenizer, None, p["question"], None)
        summary["stage2_baseline_answers"].append({"id": p["id"], "context": p["context"], "answer": text,
                                                   "broken": broken_answer(text)})
    summary["stage2_broken"] = {"unpushed_broken": sum(1 for r in summary["stage2_baseline_answers"] if r["broken"]),
                                "unpushed_total": len(summary["stage2_baseline_answers"])}
    summary["peak_rss_gb"] = round(peak_rss_gb(), 3)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    summary["stopping_rule"] = {
        "gate1_pass": summary["gate1"].get("passed", False),
        "gate2_pass": summary["gate2"]["passed"],
        "peak_rss_under_9_0_gb": summary["peak_rss_gb"] <= 9.0,
        "step_size_pass": summary["step_size"].get("passed", False),
    }
    out.mkdir(parents=True, exist_ok=True)
    # The pilot writes ONLY this summary (design l.384-388); run.json carries the same restricted facts for score213.
    write_run_json(out, "instruct", snap, model, tokenizer, {
        "stage": "pilot", "pilot_gate1": summary["gate1"], "gate2_anchors_detail": gate2,
        "tokenizer_sha256": summary["tokenizer_sha256"],
        "stage2_baseline_answers": summary["stage2_baseline_answers"], "pilot_summary": "pilot_summary.json"})
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("gate1", "throughput", "stage2_broken", "peak_rss_gb")}, indent=1))
    return 0


def pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def spearman(xs, ys) -> float:
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out

    return pearson(ranks(xs), ranks(ys))


def broken_answer(text: str) -> bool:
    """The design's mechanical broken-answer screen (design l.283-284); short valid answers are not broken."""
    import re
    from collections import Counter

    stripped = text.strip()
    if not stripped or "�" in stripped:
        return True
    words = stripped.split()
    if not any(re.search(r"[A-Za-z0-9]", w) for w in words):
        return True
    grams = Counter(tuple(w.lower() for w in words[i:i + 3]) for i in range(len(words) - 2))
    return any(c >= 4 for c in grams.values())


# ---------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("pilot", "stage1", "stage2", "fr"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--batch", type=int, default=16, help="batch size (pilot times 16 vs 32; design l.406-407)")
        p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    sub.choices["stage1"].add_argument("--with-fr", action="store_true", help="also build and run FR")
    args = ap.parse_args(argv)
    return {"pilot": cmd_pilot, "stage1": cmd_stage1, "stage2": cmd_stage2, "fr": cmd_fr}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
