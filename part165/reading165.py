#!/usr/bin/env python3
"""Part 165 reading: what gives a neuron push its extra first-step effect?

Generation-free. Rebuilds Part 165's nine arms from the registered seeds and the
real `down_proj` weights, captures hidden states on frozen natural contexts and on
the eight effect questions, measures each write's geometry, and relates that
geometry to the per-row KL Part 165's Amendment 3 effect run already stored.

Design: PART165-READING-DESIGN.md (registered before this file ran).

The direction builder below is a verbatim copy of `part165.run165._seeded_raw` /
`build_directions` (kept here so the Arc job needs no import chain); the host test
`part165/test_reading165.py` asserts the two produce identical vectors for all
nine arms.

Runs under `python -I -B reading165.py ...` from the job directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

LAYERS = tuple(range(19, 27))
SLOTS = 12
ARMS = ("n+", "n±", "k4", "k16", "k64", "k256", "k1024", "full", "iso")
K_ARMS = {"k4": 4, "k16": 16, "k64": 64, "k256": 256, "k1024": 1024}
DOSE = 0.8
DOCS = 16
SEQ = 128
SEED = 20261102
NULLS = 512
VOCAB_SAMPLE = 2000
MODEL_DEFAULT = ("E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/"
                 "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
CORPUS_DEFAULT = "E:/neuron-probe/corpus/wikitext103-train0.parquet"
READOUT_STRINGS = (" Qwen", " Qwen3", " Alibaba", " Tongyi")
PRIMARY = ("pr_w", "cos_mu", "pc16", "proj_abs_median", "mean_abs_cos_vocab")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seeded_raw(weight, layer_no, slot):
    """Part 165's registered draws, in its order (permutation, MLP, residual)."""
    width = int(weight.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(165000 + 100 * int(layer_no) + int(slot))
    permutation = torch.randperm(width, generator=generator, device="cpu")
    mlp_gaussian = torch.randn(width, generator=generator, dtype=torch.float32, device="cpu")
    hidden_gaussian = torch.randn(int(weight.shape[0]), generator=generator,
                                  dtype=torch.float32, device="cpu")
    return permutation, mlp_gaussian, hidden_gaussian


def build_directions(weight, layer_no, slot, median, pinv_cache):
    """Verbatim Part 165 construction: return {arm: (d, w)} in float32."""
    cpu_weight = weight.detach().to(device="cpu", dtype=torch.float32)
    width = int(cpu_weight.shape[1])
    permutation, gaussian, hidden_gaussian = _seeded_raw(cpu_weight, layer_no, slot)
    if int(layer_no) not in pinv_cache:
        pinv_cache[int(layer_no)] = torch.linalg.pinv(cpu_weight)
    pinv = pinv_cache[int(layer_no)]
    first = int(permutation[0].item())
    raw = {
        "n+": torch.zeros(width, dtype=torch.float32),
        "n±": torch.zeros(width, dtype=torch.float32),
        "full": gaussian.clone(),
        "iso": pinv @ hidden_gaussian,
    }
    raw["n+"][first] = 1.0
    raw["n±"][first] = 1.0 if float(gaussian[first].item()) >= 0 else -1.0
    for arm, count in K_ARMS.items():
        vector = torch.zeros(width, dtype=torch.float32)
        indices = permutation[:count]
        vector[indices] = gaussian[indices]
        raw[arm] = vector
    target = DOSE * float(median)
    out = {}
    for arm in ARMS:
        vector = raw[arm]
        write = cpu_weight @ vector
        write_norm = float(torch.linalg.vector_norm(write).item())
        if not write_norm > 0:
            raise ValueError(f"zero write norm for layer {layer_no} slot {slot} arm {arm}")
        d = vector * (target / write_norm)
        out[arm] = (d, cpu_weight @ d)
    return out


def load_model(snapshot: str, device: str):
    """The house loader (neuron_report.load_model): bf16, CausalLM then image-text.

    The checkpoint is multimodal, so the causal-LM class may refuse its config; the
    fallback is the class Part 148 runs on the Arc.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if device == "cpu":
        pass
    else:
        import transformers.modeling_utils as modeling_utils
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        kwargs["device_map"] = {"": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(snapshot, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(snapshot, **kwargs)
    if device == "cpu":
        model = model.to(device)
    model.requires_grad_(False)
    model.eval()
    return tokenizer, model


def find_text_decoder(model):
    """The decoder layer list and its parent, as Part 148 finds it.

    Qwen3.5-4B is a multimodal checkpoint: on the Arc the house loads it with
    AutoModelForImageTextToText, so the decoder is not at ``model.model``. This is
    Part 148's rule: the longest ModuleList whose blocks carry ``mlp.down_proj``,
    preferring one under a language model.
    """
    best = None
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.ModuleList) and len(module)
                and hasattr(getattr(module[0], "mlp", None), "down_proj")):
            key = ("language" in name, len(module))
            if best is None or key > best[0]:
                best = (key, name, module)
    if best is None:
        raise SystemExit("no ModuleList of blocks with mlp.down_proj found")
    _, name, modules = best
    parent_name = name.rsplit(".", 1)[0] if "." in name else ""
    parent = model.get_submodule(parent_name) if parent_name else model
    return modules, parent


@torch.no_grad()
def capture(model, decoder, ids: torch.Tensor, layers, device: str):
    """Block outputs, read from a forward hook on each decoder layer.

    The push is a pre-hook on layer L's ``down_proj``, so the residual change it
    produces appears at that block's output — the site captured here. Hooks are
    used rather than ``output_hidden_states`` because the last entry of that list
    is the final-normed tensor, not the raw last block output.
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for L in layers:
        handles.append(decoder[L].register_forward_hook(
            lambda _module, _args, output, key=L:
                captured.__setitem__(key, output[0].detach().float().cpu())))
    try:
        model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def pca(X: torch.Tensor, k: int = 16):
    """Mean and top-k principal directions of the state covariance (X is [n, d])."""
    mu = X.mean(0)
    Xc = X - mu
    covariance = Xc.mT @ Xc / X.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    k = min(k, eigenvectors.shape[1])
    eigenvalues = eigenvalues.flip(0)[:k].clamp_min(0)
    pcs = eigenvectors[:, -k:].flip(1).T.contiguous()
    return mu, pcs, eigenvalues


def rankdata(values) -> np.ndarray:
    """Average ranks, ties shared."""
    a = np.asarray(values, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ordered = a[order]
    ranks_sorted = np.empty(len(a), dtype=float)
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(a)]
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = (start + 1 + end) / 2.0
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def spearman(x, y) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def cluster_bootstrap(x, y, groups, draws: int = 10000, seed: int = 165):
    """Slot-clustered 95% interval for Spearman (rows of one slot are dependent)."""
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    unique = np.unique(groups)
    positions = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for draw in range(draws):
        pick = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([positions[g] for g in pick])
        values[draw] = spearman(x[index], y[index])
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(a.norm() * b.norm())
    return float(a @ b) / denominator if denominator > 0 else float("nan")


def geometry(w: torch.Tensor, X: torch.Tensor, mu: torch.Tensor, pcs: torch.Tensor,
             readout: dict[str, torch.Tensor], vocab_rows: torch.Tensor | None) -> dict:
    """Concentration, natural-state alignment and readout alignment of one write."""
    w = w.float()
    squared = w * w
    pr_w = float(squared.sum() ** 2 / (squared * squared).sum())
    absolute = w.abs()
    share = absolute / absolute.sum()
    ordered = torch.sort(share, descending=True).values
    mass50 = int(torch.searchsorted(ordered.cumsum(0), 0.5).item()) + 1
    projected = (X @ w) / w.norm()
    pc = (pcs @ w) / w.norm()
    row = {
        "pr_w": pr_w,
        "mass50": mass50,
        "max_share": float(share.max()),
        "cos_mu": cos(w, mu),
        "pc1": float(pc[0] ** 2),
        "pc4": float(pc[:4].pow(2).sum()),
        "pc16": float(pc.pow(2).sum()),
        "proj_median": float(projected.median()),
        "proj_q1": float(projected.quantile(0.25)),
        "proj_q3": float(projected.quantile(0.75)),
        "proj_abs_median": float(projected.abs().median()),
        "readout": {name: cos(w, covector) for name, covector in readout.items()},
    }
    if vocab_rows is not None:
        unit = w / w.norm()
        row["mean_abs_cos_vocab"] = float(
            (vocab_rows @ unit.to(vocab_rows.device)).abs().mean())
    return row


def question_geometry(w: torch.Tensor, states: torch.Tensor, mu: torch.Tensor,
                      pcs: torch.Tensor, readout: dict[str, torch.Tensor]) -> list[dict]:
    """The same alignment at each question's last prompt position."""
    w = w.float()
    unit = w / w.norm()
    rows = []
    for state in states:
        pc = (pcs @ unit)
        rows.append({
            "proj": float(state @ unit),
            "cos_state": cos(w, state),
            "cos_mu": cos(w, mu),
            "pc16": float(pc.pow(2).sum()),
            "readout": {name: cos(w, covector) for name, covector in readout.items()},
        })
    return rows


def natural_documents(args, tokenizer):
    """The frozen natural contexts: seeded wikitext rows of exactly 128 tokens."""
    if args.texts_file and Path(args.texts_file).exists():
        payload = json.loads(Path(args.texts_file).read_text(encoding="utf-8"))
        texts = payload["texts"] if isinstance(payload, dict) else payload
        return [t for t in texts if isinstance(t, str)][:args.docs], "texts-file"
    import pyarrow.parquet as pq

    table = pq.read_table(args.data, columns=["text"])
    texts = table.column("text").to_pylist()
    order = list(range(len(texts)))
    import random as _random
    _random.Random(args.seed).shuffle(order)
    chosen = []
    for index in order:
        raw = texts[index]
        if not isinstance(raw, str) or not raw.strip():
            continue
        encoded = tokenizer(raw, truncation=True, max_length=SEQ)["input_ids"]
        if len(encoded) != SEQ:
            continue
        chosen.append(raw)
        if len(chosen) == args.docs:
            return chosen, "parquet"
    raise RuntimeError(f"only {len(chosen)} rows tokenise to {SEQ} tokens")


def load_effects(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") != "effect":
                continue
            key = (int(row["question_index"]), int(row["layer"]),
                   int(row["slot"]), row["arm"])
            rows[key] = {"kl": float(row["kl"]), "rel_change": float(row["rel_change"])}
    return rows


def load_norms(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(int(r["layer"]), int(r["slot"]), r["arm"]): float(r["write_norm"])
            for r in payload["directions"]}


def frozen_readout(tokenizer, parent, head_weight, vocab_sample: int):
    """Single-token strings become covectors gamma * u_t; multi-token ones are dropped."""
    device = head_weight.device
    gamma = parent.norm.weight.detach().float()
    embedding = head_weight
    covectors, used, dropped = {}, {}, []
    for text in READOUT_STRINGS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            dropped.append({"string": text, "tokens": ids})
            continue
        token = int(ids[0])
        covectors[text] = (gamma * embedding[token].float()).cpu()
        used[text] = token
    generator = torch.Generator().manual_seed(SEED)
    sample = torch.randint(0, embedding.shape[0], (vocab_sample,), generator=generator)
    rows = embedding[sample.to(device)].float()
    rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return covectors, used, dropped, rows, gamma


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--norms", default="results/part165/norms.json")
    parser.add_argument("--effects", default="results/part165/effects.jsonl")
    parser.add_argument("--medians", default="results/part112/medians.json")
    parser.add_argument("--data", default=CORPUS_DEFAULT)
    parser.add_argument("--texts-file", default="part165/reading-texts.json",
                        help="the frozen natural contexts (a JSON list, or an object with 'texts')")
    parser.add_argument("--out", default="results/part165/reading")
    parser.add_argument("--docs", type=int, default=DOCS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    parser.add_argument("--slots", type=int, default=SLOTS)
    parser.add_argument("--questions", default="part165/reading-questions.json")
    parser.add_argument("--vocab-sample", type=int, default=VOCAB_SAMPLE)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--nulls", type=int, default=NULLS)
    args = parser.parse_args(argv)

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    norms = load_norms(Path(args.norms))
    effects = load_effects(Path(args.effects))
    medians = {int(k): float(v) for k, v in
               json.loads(Path(args.medians).read_text(encoding="utf-8")).items()}
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]

    tokenizer, model = load_model(args.model, args.device)
    decoder, text_model = find_text_decoder(model)
    embed_weight = getattr(text_model, "embed_tokens", None)
    embed_weight = embed_weight.weight if embed_weight is not None else \
        model.get_input_embeddings().weight
    head = getattr(model, "lm_head", None)
    head_weight = head.weight if head is not None else embed_weight
    if tuple(head_weight.shape) != tuple(embed_weight.shape):
        head_weight = embed_weight
    texts, context_source = natural_documents(args, tokenizer)
    if len(texts) != args.docs:
        parser.error(f"wanted {args.docs} documents, got {len(texts)}")
    question_ids = [tokenizer(q, return_tensors="pt")["input_ids"][0] for q in questions]

    joined = "\n".join(texts)
    disjoint = all(q not in joined for q in questions) and not any(
        text in q for text in texts for q in questions)
    gates = {"directions": True, "effects": True, "contexts": bool(disjoint),
             "measures": True, "null_validity": True, "notes": []}
    if not disjoint:
        gates["notes"].append("a natural document overlaps a Part 165 question")

    vocab_rows = None
    readout, readout_ids, readout_dropped, vocab_rows, _gamma = frozen_readout(
        tokenizer, text_model, head_weight, args.vocab_sample)

    document_states = [capture(model, decoder,
                               tokenizer(text, return_tensors="pt")["input_ids"][0],
                               layers, args.device) for text in texts]
    question_captures = [capture(model, decoder, ids, layers, args.device)
                         for ids in question_ids]
    state_dim = int(document_states[0][layers[0]].shape[1])
    natural = {L: torch.stack([state[L] for state in document_states]).reshape(-1, state_dim)
               for L in layers}
    question_states = {L: torch.stack([state[L][-1] for state in question_captures])
                       for L in layers}

    pinv_cache = {}
    slot_rows, null_rows = [], {}
    weights = {L: decoder[L].mlp.down_proj.weight for L in layers}
    for L in layers:
        X = natural[L].float()
        mu, pcs, eigenvalues = pca(X, 16)
        gram = pcs @ pcs.mT
        orthonormal = bool(torch.allclose(gram, torch.eye(16), atol=1e-6))
        gates["measures"] = gates["measures"] and orthonormal and bool(torch.isfinite(X).all())
        if not orthonormal:
            gates["notes"].append(f"layer {L}: PCs not orthonormal")
        generator = torch.Generator().manual_seed(SEED)
        directions = torch.randn(args.nulls, int(X.shape[1]), generator=generator)
        directions = directions / directions.norm(dim=1, keepdim=True)
        null = []
        for u in directions:
            row = geometry(u, X, mu, pcs, readout, None)
            row["proj_abs_median"] = float(((X @ u) / u.norm()).abs().median())
            null.append(row)
        null_rows[L] = null

        for slot in range(args.slots):
            built = build_directions(weights[L], L, slot, medians[L], pinv_cache)
            arms = {}
            for arm in ARMS:
                d, w = built[arm]
                recorded = norms.get((L, slot, arm))
                relative = abs(float(w.norm()) - recorded) / recorded if recorded else float("inf")
                if not (recorded is not None and relative <= 1e-4):
                    gates["directions"] = False
                    gates["notes"].append(
                        f"write norm mismatch L{L} S{slot} {arm}: {relative:.2e}")
                row = geometry(w, X, mu, pcs, readout, vocab_rows)
                squared = d * d
                row.update({
                    "write_norm": float(w.norm()),
                    "norm_relative_error": relative,
                    "pr_d": float(squared.sum() ** 2 / (squared * squared).sum()),
                    "nnz_d": int((d != 0).sum()),
                    "questions": question_geometry(w, question_states[L], mu, pcs, readout),
                })
                arms[arm] = row
            slot_rows.append({"layer": L, "slot": slot, "arms": arms})

        iso = [r["arms"]["iso"] for r in slot_rows if r["layer"] == L]
        for key in ("pr_w", "cos_mu", "proj_abs_median"):
            band = np.percentile([row[key] for row in null_rows[L]], [5, 95])
            median = float(np.median([row[key] for row in iso]))
            if not (band[0] - 1e-9 <= median <= band[1] + 1e-9):
                gates["null_validity"] = False
                gates["notes"].append(
                    f"layer {L}: iso median {median:.4g} outside the null band {band} on {key}")

    # Join the effect rows: one row per (question, layer, slot, arm).
    records = []
    for row in slot_rows:
        for arm, values in row["arms"].items():
            for q_index, q in enumerate(values["questions"]):
                key = (q_index, row["layer"], row["slot"], arm)
                stored = effects.get(key)
                if stored is None:
                    gates["effects"] = False
                    continue
                records.append({"question": q_index, "layer": row["layer"],
                                "slot": row["slot"], "arm": arm, "kl": stored["kl"],
                                "rel_change": stored["rel_change"], **{
                                    k: values[k] for k in
                                    ("pr_w", "mass50", "max_share", "cos_mu", "pc1",
                                     "pc4", "pc16", "proj_abs_median",
                                     "mean_abs_cos_vocab")},
                                "proj": q["proj"], "readout_qwen": q["readout"].get(" Qwen")})

    def array(rows, key):
        return np.array([r[key] if r[key] is not None else np.nan for r in rows], dtype=float)

    correlations = {"within_arm": {}, "across_arms": {}}
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        if len(rows) < 8:
            continue
        groups = [r["slot"] for r in rows]
        entry = {}
        for key in PRIMARY + ("proj", "readout_qwen", "mass50", "max_share", "pc1", "pc4"):
            x, y = array(rows, key), array(rows, "kl")
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 8:
                continue
            rho = spearman(x[mask], y[mask])
            ci = cluster_bootstrap(x[mask], y[mask], np.array(groups)[mask],
                                   draws=args.bootstrap) if arm in ("n+", "full") else None
            entry[key] = {"rho": rho, "n": int(mask.sum()), "ci95": ci}
        correlations["within_arm"][arm] = entry

    per_layer = {}
    for L in layers:
        rows = [r for r in records if r["layer"] == L]
        means = {}
        for arm in ARMS:
            arm_rows = [r for r in rows if r["arm"] == arm]
            if not arm_rows:
                continue
            means[arm] = {"kl": float(np.mean([r["kl"] for r in arm_rows])),
                          "pr_w": float(np.mean([r["pr_w"] for r in arm_rows])),
                          "proj_abs_median": float(np.mean([r["proj_abs_median"] for r in arm_rows])),
                          "cos_mu": float(np.mean([r["cos_mu"] for r in arm_rows]))}
        arms = [a for a in ARMS if a in means]
        per_layer[L] = {
            "arms": means,
            "rho_pr_kl": spearman([means[a]["pr_w"] for a in arms],
                                  [means[a]["kl"] for a in arms]),
        }
    correlations["across_arms"] = per_layer

    # R5: transport the within-n+ slope to the n+ - full gap.
    n_plus = [r for r in records if r["arm"] == "n+"]
    x = np.log(array(n_plus, "pr_w"))
    y = np.log(array(n_plus, "kl"))
    mask = np.isfinite(x) & np.isfinite(y)
    slope = float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() > 8 else float("nan")
    means_l = {arm: {"log_kl": float(np.mean([math.log(r["kl"]) for r in records
                                              if r["arm"] == arm and r["kl"] > 0])),
                     "log_pr": float(np.mean([math.log(r["pr_w"]) for r in records
                                              if r["arm"] == arm]))}
               for arm in ARMS if any(r["arm"] == arm for r in records)}
    raw_gap = means_l["n+"]["log_kl"] - means_l["full"]["log_kl"]
    predicted_gap = slope * (means_l["n+"]["log_pr"] - means_l["full"]["log_pr"])
    explained = predicted_gap / raw_gap if raw_gap else float("nan")

    def layer_rho(key, arm):
        values = []
        for L in layers:
            rows = [r for r in records if r["arm"] == arm and r["layer"] == L]
            if len(rows) <= 8:
                continue
            values.append(spearman(array(rows, key), array(rows, "kl")))
        return values

    n_plus_layer = layer_rho("pr_w", "n+")
    bars = {
        "R1": {"rule": "n+ pooled rho(pr_w, kl) <= -0.30 and <= -0.15 in >=5/8 layers",
               "rho": correlations["within_arm"].get("n+", {}).get("pr_w", {}).get("rho"),
               "layers_at_least_-0.15": int(sum(r <= -0.15 for r in n_plus_layer)),
               "layers": n_plus_layer,
               "pass": bool(correlations["within_arm"].get("n+", {}).get("pr_w", {}).get("rho", 0) <= -0.30
                            and sum(r <= -0.15 for r in n_plus_layer) >= 5)},
        "R2": {"rule": "per-layer rho(arm-mean pr_w, arm-mean kl) <= -0.70 in >=6/8 layers",
               "rhos": {L: per_layer[L]["rho_pr_kl"] for L in layers},
               "layers_at_most_-0.70": int(sum(v <= -0.70 for v in
                                              (per_layer[L]["rho_pr_kl"] for L in layers))),
               "pass": bool(sum(v <= -0.70 for v in
                                (per_layer[L]["rho_pr_kl"] for L in layers)) >= 6)},
        "R3": {"rule": "median |proj| n+ >= 1.5x the isotropic null median in >=6/8 layers",
               "ratios": {}, "pass": None},
        "R4": {"rule": "|rho(readout Qwen, kl)| < 0.30 within n+",
               "rho": correlations["within_arm"].get("n+", {}).get("readout_qwen", {}).get("rho"),
               "pass": None},
        "R5": {"rule": "within-n+ slope explains >=50% of the n+ - full log-KL gap",
               "slope": slope, "raw_gap": raw_gap, "predicted_gap": predicted_gap,
               "explained": explained, "pass": bool(explained >= 0.5) if math.isfinite(explained) else False},
    }
    ratios = {}
    for L in layers:
        rows = [r for r in records if r["arm"] == "n+" and r["layer"] == L]
        null = [row["proj_abs_median"] for row in null_rows[L]]
        ratios[L] = float(np.median([r["proj_abs_median"] for r in rows]) / np.median(null))
    bars["R3"]["ratios"] = ratios
    bars["R3"]["pass"] = bool(sum(v >= 1.5 for v in ratios.values()) >= 6)
    bars["R4"]["pass"] = bool(abs(bars["R4"]["rho"]) < 0.30) if bars["R4"]["rho"] is not None else False

    gates["passed"] = all(gates[k] for k in ("directions", "effects", "contexts",
                                             "measures", "null_validity"))

    result = {
        "part": 165,
        "kind": "reading",
        "design": "PART165-READING-DESIGN.md",
        "config": {
            "model": args.model, "device": args.device, "layers": list(layers),
            "slots": args.slots, "docs": args.docs, "seed": args.seed,
            "context_source": context_source, "nulls": args.nulls,
            "vocab_sample": args.vocab_sample, "questions": questions,
            "readout_strings": list(readout_ids), "readout_dropped": readout_dropped,
            "seconds": round(time.time() - started, 1),
            "inputs_sha256": {name: sha256_file(Path(path)) for name, path in
                              (("norms", args.norms), ("effects", args.effects),
                               ("medians", args.medians))},
        },
        "gates": gates,
        "nulls": {L: {"pr_w_median": float(np.median([r["pr_w"] for r in rows])),
                      "cos_mu_median": float(np.median([r["cos_mu"] for r in rows])),
                      "proj_abs_median": float(np.median([r["proj_abs_median"] for r in rows]))}
                  for L, rows in null_rows.items()},
        "slots": slot_rows,
        "correlations": correlations,
        "bars": bars,
        "predictions": {k: bars[k]["pass"] for k in bars},
    }
    (out_dir / "reading.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    lines = ["# Part 165 reading", "",
             f"gates: **{'PASS' if gates['passed'] else 'FAIL'}** "
             f"(directions {gates['directions']}, effects {gates['effects']}, "
             f"contexts {gates['contexts']}, measures {gates['measures']}, "
             f"null validity {gates['null_validity']})", ""]
    if gates["notes"]:
        lines += ["- " + note for note in gates["notes"]] + [""]
    lines += ["## Bars", "", "| bar | rule | result |", "| --- | --- | --- |"]
    for name, bar in bars.items():
        lines.append(f"| {name} | {bar['rule']} | {'**pass**' if bar['pass'] else 'fail'} |")
    lines += ["", "## Within-arm Spearman against KL", "",
              "| arm | PR(w) | cos(w, mu) | PC16 | |proj| | readout |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in ARMS:
        entry = correlations["within_arm"].get(arm)
        if not entry:
            continue
        cells = []
        for key in ("pr_w", "cos_mu", "pc16", "proj_abs_median", "readout_qwen"):
            value = entry.get(key, {}).get("rho")
            cells.append("–" if value is None else f"{value:.3f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    lines += ["", "## Across arms", "",
              "| layer | rho(PR(w), KL) |", "| --- | ---: |"]
    for L in layers:
        lines.append(f"| {L} | {per_layer[L]['rho_pr_kl']:.3f} |")
    lines += ["", "## Gap", "",
              f"within-n+ slope {slope:.3f}; raw n+ - full log-KL gap {raw_gap:.3f}; "
              f"predicted from PR {predicted_gap:.3f}; explained {explained:.0%}.", ""]
    (out_dir / "reading-tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"gates {'PASS' if gates['passed'] else 'FAIL'}; "
          f"bars " + ", ".join(f"{k}={'pass' if v['pass'] else 'fail'}" for k, v in bars.items()))
    print(f"wrote {out_dir/'reading.json'} and reading-tables.md in {result['config']['seconds']}s")
    return 0 if gates["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
