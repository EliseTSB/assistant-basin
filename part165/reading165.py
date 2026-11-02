#!/usr/bin/env python3
"""Part 165 reading: what gives a neuron push its extra first-step effect?

Generation-free. Rebuilds Part 165's nine arms from the registered seeds and the
real `down_proj` weights, captures hidden states on frozen natural contexts and on
the eight effect questions at the step the effect run measured, measures each
write's geometry, and relates that geometry to the per-row KL stored by Part 165's
Amendment 3 effect run.

Design and Amendment 1: PART165-READING-DESIGN.md (registered before this ran;
Amendment 1 folds the 2026-11-02 cold read).

House paths, as the Arc has run them: `neuron_report.load_model` (dtype=, CausalLM
then ImageTextToText) and `neuron_report.build_prompt` (chat template,
add_generation_prompt, enable_thinking=False).

The direction builder below is a verbatim copy of `part165.run165._seeded_raw` /
`build_directions`; the host test asserts identical vectors for all nine arms.

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

import neuron_report as nr  # noqa: E402  (copied into the job; house loader and prompt)

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
READOUT_STRINGS = (" Alibaba", " Qwen", " Qwen3", " Tongyi")
READOUT_KEY = " Alibaba"  # the only maker string that is a single token (Amendment 1, R4)
PRIMARY = ("pr_w", "cos_mu", "pc16", "proj_abs_median", "mean_abs_cos_vocab")
# R4's clustered 95% interval is promised by Amendment 1; ``readout`` is not a
# primary geometry measure but it is bootstrapped (recheck N2).
BOOTSTRAP_KEYS = PRIMARY + ("readout",)
BARS = ("R1", "R2", "R3", "R4", "R5")


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


def find_text_decoder(model):
    """The decoder layer list and its parent, as Part 148 finds it.

    Qwen3.5-4B is multimodal, so the decoder is not necessarily at ``model.model``.
    ``neuron_report.load_model`` falls back to AutoModelForImageTextToText when the
    CausalLM class refuses the config; on transformers 5.20 the CausalLM class
    accepts it and the fallback is never reached (recheck N6). Longest ModuleList
    whose blocks carry ``mlp.down_proj``, preferring a language model.
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


def _layer_tensor(output):
    """A decoder layer returns a tensor in 5.20; accept a tuple as well."""
    value = output[0] if isinstance(output, (tuple, list)) else output
    return value


@torch.no_grad()
def capture(model, decoder, ids: torch.Tensor, layers, device: str):
    """Block outputs, read from a forward hook on each decoder layer.

    The push is a pre-hook on layer L's ``down_proj``, so the residual change it
    produces appears at that block's output. Hooks are used rather than
    ``output_hidden_states`` because the last entry of that list is final-normed.
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for L in layers:
        handles.append(decoder[L].register_forward_hook(
            lambda _module, _args, output, key=L:
                captured.__setitem__(
                    key, _layer_tensor(output)[0].detach().float().cpu())))
    try:
        model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def encode_documents(texts, tokenizer, seq: int = SEQ):
    """The registered population: the first ``seq`` tokens of each selected row.

    Part 165's reading froze the row *texts*; the selection tested them with
    truncation, so the job must truncate too, and fail loudly rather than stack
    ragged tensors (cold read F1).
    """
    encoded = []
    for index, text in enumerate(texts):
        ids = tokenizer(text, truncation=True, max_length=seq,
                        add_special_tokens=False)["input_ids"]
        if len(ids) != seq:
            raise ValueError(
                f"document {index} has {len(ids)} tokens after truncation, expected {seq}")
        encoded.append(torch.tensor(ids, dtype=torch.long))
    return encoded


@torch.no_grad()
def effect_question_state(model, decoder, tokenizer, question: str, layers, device: str):
    """The state the effect run measured: prompt + first greedy token, last position.

    ``common86.token_inputs`` renders the chat prompt with
    ``neuron_report.build_prompt``; ``AddRowsHook`` leaves prefill untouched, so the
    first token is plain, and the measured scores are the second step's, where the
    push acts (cold read F4).
    """
    prompt = nr.build_prompt(tokenizer, question, False)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    logits = model(input_ids=ids.to(device), use_cache=False).logits
    first = int(logits[0, -1].argmax().item())
    extended = torch.cat([ids[0].cpu(), torch.tensor([first], dtype=torch.long)])
    return capture(model, decoder, extended, layers, device), first


def pca(X: torch.Tensor, k: int = 16):
    """Mean and top-k principal directions, in float64 on the CPU (cold read F6)."""
    Xd = X.double()
    mu = Xd.mean(0)
    Xc = Xd - mu
    covariance = Xc.mT @ Xc / Xd.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    k = min(k, eigenvectors.shape[1])
    top = eigenvalues.flip(0)[:k]
    pcs = eigenvectors[:, -k:].flip(1).T.contiguous().float()
    return mu.float(), pcs, top, float(eigenvalues.min()), float(eigenvalues.max())


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
    """Cluster bootstrap over ``(layer, slot)`` groups, one-way.

    Rows of one slot are dependent, so a slot-only interval understates
    uncertainty (cold read F8/F15). Questions repeat across slots and are not a
    separate resampling dimension in this estimator; the docstring said "two-way"
    and the code never was (recheck N8).
    """
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
    """The same alignment at each question's measured step."""
    w = w.float()
    unit = w / w.norm()
    rows = []
    for state in states:
        pc = pcs @ unit
        rows.append({
            "proj": float(state @ unit),
            "cos_state": cos(w, state),
            "cos_mu": cos(w, mu),
            "pc16": float(pc.pow(2).sum()),
            "readout": {name: cos(w, covector) for name, covector in readout.items()},
        })
    return rows


def load_questions(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload
    if len(questions) != 8:
        raise ValueError(f"expected 8 frozen questions, got {len(questions)}")
    return questions


def load_texts(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    texts = payload["texts"] if isinstance(payload, dict) else payload
    return payload if isinstance(payload, dict) else {}, texts


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
    """Single-token strings become covectors (1 + gamma) * u_t; others are dropped.

    Qwen3_5RMSNorm multiplies by ``1 + weight`` (cold read F10); the weight is
    zero-initialised, so ``gamma`` alone is wrong by a per-coordinate gain.
    """
    device = head_weight.device
    gamma = 1.0 + parent.norm.weight.detach().float()
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
    return covectors, used, dropped, rows


@torch.no_grad()
def capture_check(model, decoder, ids, layers, device):
    """Each hooked block output must equal the model's own hidden_states entry."""
    captured = capture(model, decoder, ids, layers, device)
    out = model(input_ids=ids.unsqueeze(0).to(device), output_hidden_states=True,
                use_cache=False)
    hidden = out.hidden_states
    errors = {}
    last = len(decoder) - 1
    for L in layers:
        if L >= last:  # the final entry of hidden_states is normed
            continue
        reference = hidden[L + 1][0].float().to(captured[L].device)
        scale = float(reference.abs().max()) or 1.0
        errors[L] = float((captured[L] - reference).abs().max()) / scale
    return errors, max(errors.values()) if errors else float("nan")


@torch.no_grad()
def gap_ols(rows):
    """The registered R5 estimator: ``log kl ~ log PR + layer fixed effects``.

    The design carries an explicit intercept and a dummy for every layer but the
    lowest, so layer 19 is the reference level rather than being pinned to the
    slope line. Without the constant the slope is biased (recheck N1): on the
    synthetic control it read -0.366 against -0.401.
    """
    log_pr = np.log(np.array([r["pr_w"] for r in rows], dtype=float))
    log_kl = np.log(np.array([r["kl"] for r in rows], dtype=float))
    elements = sorted({r["layer"] for r in rows})
    columns = [log_pr, np.ones(len(rows), dtype=float)]
    for L in elements[1:]:
        columns.append(np.array([1.0 if r["layer"] == L else 0.0 for r in rows]))
    beta, *_ = np.linalg.lstsq(np.column_stack(columns), log_kl, rcond=None)
    return float(beta[0]), beta


def bar_label(bar) -> str:
    """Three-way verdict: a bar that was not measured is not a failure (N4)."""
    value = bar.get("pass")
    if value is True:
        return "**pass**"
    return "fail" if value is False else "not measured"


def site_check(model, decoder, layer: int, d: torch.Tensor, w: torch.Tensor,
               ids: torch.Tensor, device: str):
    """A push at down_proj must change the block output by W d (cold read F5)."""
    def last_position(output):
        return _layer_tensor(output)[0, -1].detach().float().cpu()

    plain = {}
    handle = decoder[layer].register_forward_hook(
        lambda _m, _a, output: plain.__setitem__("h", last_position(output)))
    model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    handle.remove()

    def push(_module, args):
        value = args[0].clone()
        value[0, -1] = value[0, -1] + d.to(value.device, value.dtype)
        return (value,) + tuple(args[1:])

    pushed = {}
    pre = decoder[layer].mlp.down_proj.register_forward_pre_hook(push)
    handle = decoder[layer].register_forward_hook(
        lambda _m, _a, output: pushed.__setitem__("h", last_position(output)))
    model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    handle.remove()
    pre.remove()
    delta = pushed["h"] - plain["h"]
    return float((delta - w).norm() / w.norm())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--norms", default="results/part165/norms.json")
    parser.add_argument("--effects", default="results/part165/effects.jsonl")
    parser.add_argument("--medians", default="results/part112/medians.json")
    parser.add_argument("--texts-file", default="part165/reading-texts.json")
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
    questions = load_questions(Path(args.questions))
    texts_meta, texts = load_texts(Path(args.texts_file))

    tokenizer, model = nr.load_model(args.model, None, args.device)
    decoder, text_model = find_text_decoder(model)
    embed_weight = getattr(text_model, "embed_tokens", None)
    embed_weight = embed_weight.weight if embed_weight is not None else \
        model.get_input_embeddings().weight
    head = getattr(model, "lm_head", None)
    head_weight = head.weight if head is not None else embed_weight
    if tuple(head_weight.shape) != tuple(embed_weight.shape):
        head_weight = embed_weight
    t_texts = time.time()
    documents = encode_documents(texts, tokenizer)
    gates = {"directions": True, "effects": True, "contexts": True,
             "measures": True, "capture": True, "notes": []}
    if len(documents) != args.docs:
        gates["contexts"] = False
        gates["notes"].append(f"wanted {args.docs} documents, got {len(documents)}")
    if "corpus_sha256" not in texts_meta or "rows" not in texts_meta:
        gates["contexts"] = False
        gates["notes"].append("texts file carries no corpus_sha256/rows provenance")
    joined = "\n".join(texts)
    if any(q in joined for q in questions) or any(t in q for t in texts for q in questions):
        gates["contexts"] = False
        gates["notes"].append("a natural document overlaps a Part 165 question")
    for key, stored in effects.items():
        if not (math.isfinite(stored["kl"]) and math.isfinite(stored["rel_change"])):
            gates["effects"] = False
            gates["notes"].append(f"non-finite effect row {key}")

    readout, readout_ids, readout_dropped, vocab_rows = frozen_readout(
        tokenizer, text_model, head_weight, args.vocab_sample)
    if READOUT_KEY not in readout:
        gates["measures"] = False
        gates["notes"].append(f"readout key {READOUT_KEY!r} is not a single token")

    print(f"documents encoded in {time.time() - t_texts:.1f}s; "
          f"readout {sorted(readout_ids)}; dropped {[d['string'] for d in readout_dropped]}",
          flush=True)

    document_states = [capture(model, decoder, ids, layers, args.device) for ids in documents]
    state_dim = int(document_states[0][layers[0]].shape[1])
    natural = {L: torch.stack([state[L] for state in document_states]).reshape(-1, state_dim)
               for L in layers}

    # Capture site: hooked outputs equal the model's own; a push moves the block
    # output by exactly W d. Both can fail; the iso-vs-null check could not (F5).
    errors, worst = capture_check(model, decoder, documents[0], layers, args.device)
    # Relative to the state norm, and at the bf16 rounding floor rather than at
    # exact equality: two forward passes need not be bit-identical on XPU (N7).
    if not (worst <= 1e-2):
        gates["capture"] = False
        gates["notes"].append(f"capture mismatch {worst:.2e} relative: {errors}")
    pinv_cache = {}
    first_layer = layers[0]
    first_d, first_w = build_directions(
        decoder[first_layer].mlp.down_proj.weight, first_layer, 0, medians[first_layer],
        pinv_cache)["n+"]
    site = site_check(model, decoder, first_layer, first_d, first_w, documents[0], args.device)
    # A bf16 block-output delta rounds at ~4e-3 relative, max 5.4e-3 with heavy
    # residual dims, plus ~1e-3 for casting d; 5e-3 was inside that floor (N3).
    if not (site <= 1e-2):
        gates["capture"] = False
        gates["notes"].append(f"site check L{first_layer}: |delta - W d| / |W d| = {site:.2e}")

    question_states, first_tokens = {}, {}
    for q_index, question in enumerate(questions):
        captured, first = effect_question_state(
            model, decoder, tokenizer, question, layers, args.device)
        for L in layers:
            question_states.setdefault(L, []).append(captured[L][-1])
        first_tokens[q_index] = first
    question_states = {L: torch.stack(rows) for L, rows in question_states.items()}

    slot_rows, null_rows = [], {}
    weights = {L: decoder[L].mlp.down_proj.weight for L in layers}
    for L in layers:
        X = natural[L].float()
        mu, pcs, eigenvalues, eig_min, eig_max = pca(X, 16)
        orthonormal = float((pcs @ pcs.mT - torch.eye(pcs.shape[0])).abs().max())
        if not (eig_min >= -1e-8 * max(eig_max, 1e-12) and orthonormal <= 1e-5
                and bool(torch.isfinite(X).all())):
            gates["measures"] = False
            gates["notes"].append(
                f"layer {L}: eig_min {eig_min:.3e}, eig_max {eig_max:.3e}, "
                f"orthonormality {orthonormal:.2e}")
        generator = torch.Generator().manual_seed(SEED)
        directions = torch.randn(args.nulls, int(X.shape[1]), generator=generator)
        directions = directions / directions.norm(dim=1, keepdim=True)
        null = [geometry(u, X, mu, pcs, readout, vocab_rows) for u in directions]
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
        pinv_cache.clear()  # ~94 MB per layer; only the current layer needs it (N5)

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
                                "proj": q["proj"],
                                "readout": q["readout"].get(READOUT_KEY)})

    gates["passed"] = all(gates[k] for k in
                          ("directions", "effects", "contexts", "measures", "capture"))

    result = {
        "part": 165,
        "kind": "reading",
        "design": "PART165-READING-DESIGN.md",
        "amendments": ["Amendment 1 (2026-11-02, cold read)",
                       "Amendment 2 (2026-11-02, recheck of c7c0290)"],
        "config": {
            "model": args.model, "device": args.device, "layers": list(layers),
            "slots": args.slots, "docs": args.docs, "seed": args.seed,
            "population": "first 128 tokens of each frozen row", "nulls": args.nulls,
            "vocab_sample": args.vocab_sample, "questions": questions,
            "first_greedy_tokens": first_tokens,
            "readout_strings": list(readout_ids), "readout_dropped": readout_dropped,
            "readout_key": READOUT_KEY,
            "capture_max_relative_error": worst, "site_relative_error": site,
            "seconds": round(time.time() - started, 1),
            "inputs_sha256": {name: sha256_file(Path(path)) for name, path in
                              (("norms", args.norms), ("effects", args.effects),
                               ("medians", args.medians), ("texts", args.texts_file),
                               ("questions", args.questions))},
        },
        "gates": gates,
        "nulls": {L: {"pr_w_median": float(np.median([r["pr_w"] for r in rows])),
                      "cos_mu_median": float(np.median([r["cos_mu"] for r in rows])),
                      "proj_abs_median": float(np.median([r["proj_abs_median"] for r in rows])),
                      "mean_abs_cos_vocab_median":
                          float(np.median([r["mean_abs_cos_vocab"] for r in rows]))}
                  for L, rows in null_rows.items()},
        "slots": slot_rows,
    }

    if not gates["passed"]:
        result["bars"] = {name: {"pass": None, "reason": "gate failed"} for name in BARS}
        result["correlations"] = {}
    else:
        def array(rows, key):
            return np.array([r[key] if r[key] is not None else np.nan for r in rows], dtype=float)

        correlations = {"within_arm": {}, "across_arms": {}}
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm]
            if len(rows) < 8:
                continue
            groups = np.array([f"{r['layer']}:{r['slot']}" for r in rows])
            entry = {}
            for key in BOOTSTRAP_KEYS + ("proj", "mass50", "max_share", "pc1", "pc4"):
                x, y = array(rows, key), array(rows, "kl")
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 8:
                    continue
                rho = spearman(x[mask], y[mask])
                ci = cluster_bootstrap(x[mask], y[mask], groups[mask],
                                       draws=args.bootstrap) if key in BOOTSTRAP_KEYS else None
                entry[key] = {"rho": rho, "n": int(mask.sum()), "ci95": ci,
                              "n_clusters": int(len(np.unique(groups[mask])))}
            correlations["within_arm"][arm] = entry

        per_layer = {}
        for L in layers:
            means = {}
            for arm in ARMS:
                arm_rows = [r for r in records if r["arm"] == arm and r["layer"] == L]
                if arm_rows:
                    means[arm] = {"kl": float(np.mean([r["kl"] for r in arm_rows])),
                                  "pr_w": float(np.mean([r["pr_w"] for r in arm_rows])),
                                  "proj_abs_median": float(np.mean(
                                      [r["proj_abs_median"] for r in arm_rows])),
                                  "cos_mu": float(np.mean([r["cos_mu"] for r in arm_rows]))}
            present = [a for a in ARMS if a in means]
            per_layer[L] = {
                "arms": means,
                "rho_pr_kl": spearman([means[a]["pr_w"] for a in present],
                                      [means[a]["kl"] for a in present]),
            }
        correlations["across_arms"] = per_layer

        # R5, the registered estimator: log kl ~ log PR + layer fixed effects.
        usable = [r for r in records if r["kl"] > 0]
        slope, coefficients = gap_ols(usable)
        mean_log_pr = {arm: float(np.mean([math.log(r["pr_w"]) for r in usable
                                           if r["arm"] == arm]))
                       for arm in ARMS if any(r["arm"] == arm for r in usable)}
        mean_log_kl = {arm: float(np.mean([math.log(r["kl"]) for r in usable
                                           if r["arm"] == arm]))
                       for arm in ARMS if any(r["arm"] == arm for r in usable)}
        raw_gap = mean_log_kl["n+"] - mean_log_kl["full"]
        predicted_gap = slope * (mean_log_pr["n+"] - mean_log_pr["full"])
        explained = predicted_gap / raw_gap if raw_gap else float("nan")

        def layer_rho(key, arm):
            values = []
            for L in layers:
                rows = [r for r in records if r["arm"] == arm and r["layer"] == L]
                if len(rows) > 8:
                    values.append(spearman(array(rows, key), array(rows, "kl")))
            return values

        n_plus_layer = layer_rho("pr_w", "n+")
        within_n = correlations["within_arm"].get("n+", {})
        r1 = within_n.get("pr_w", {})
        r4 = within_n.get("readout", {})
        ratios = {}
        for L in layers:
            rows = [r for r in records if r["arm"] == "n+" and r["layer"] == L]
            ratios[L] = float(np.median([r["proj_abs_median"] for r in rows])
                              / np.median([row["proj_abs_median"] for row in null_rows[L]]))
        bars = {
            "R1": {"rule": "n+ pooled rho(pr_w, kl) <= -0.30 and <= -0.15 in >=5/8 layers",
                   "rho": r1.get("rho"), "ci95": r1.get("ci95"),
                   "layers_at_least_-0.15": int(sum(v <= -0.15 for v in n_plus_layer)),
                   "layers": n_plus_layer,
                   "pass": bool((r1.get("rho") is not None and r1["rho"] <= -0.30)
                                and sum(v <= -0.15 for v in n_plus_layer) >= 5)},
            "R2": {"rule": "per-layer rho(arm-mean pr_w, arm-mean kl) <= -0.70 in >=6/8 layers",
                   "rhos": {L: per_layer[L]["rho_pr_kl"] for L in layers},
                   "layers_at_most_-0.70": int(sum(v <= -0.70 for v in
                                                  (per_layer[L]["rho_pr_kl"] for L in layers))),
                   "pass": bool(sum(v <= -0.70 for v in
                                    (per_layer[L]["rho_pr_kl"] for L in layers)) >= 6)},
            "R3": {"rule": "median |proj| n+ >= 1.5x the isotropic null median in >=6/8 layers",
                   "ratios": ratios, "pass": bool(sum(v >= 1.5 for v in ratios.values()) >= 6)},
            "R4": {"rule": "(1+gamma)*u_Alibaba: |rho(readout, kl)| < 0.30 within n+",
                   "rho": r4.get("rho"), "ci95": r4.get("ci95"),
                   "n_effective": r4.get("n_clusters"),
                   "pass": None if r4.get("rho") is None else bool(abs(r4["rho"]) < 0.30)},
            "R5": {"rule": "OLS log kl ~ log PR + layer FE explains >=50% of the n+ - full gap",
                   "slope": slope, "raw_gap": raw_gap, "predicted_gap": predicted_gap,
                   "explained": explained,
                   "pass": bool(explained >= 0.5) if math.isfinite(explained) else False},
        }
        result["correlations"] = correlations
        result["bars"] = bars

    (out_dir / "reading.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    lines = ["# Part 165 reading", "",
             f"gates: **{'PASS' if gates['passed'] else 'FAIL'}** "
             f"(directions {gates['directions']}, effects {gates['effects']}, "
             f"contexts {gates['contexts']}, measures {gates['measures']}, "
             f"capture {gates['capture']})", ""]
    if gates["notes"]:
        lines += ["- " + note for note in gates["notes"]] + [""]
    if gates["passed"]:
        lines += ["## Bars", "", "| bar | rule | result |", "| --- | --- | --- |"]
        for name, bar in result["bars"].items():
            lines.append(f"| {name} | {bar['rule']} | "
                         f"{bar_label(bar)} |")
        lines += ["", "## Within-arm Spearman against KL", "",
                  "| arm | PR(w) | cos(w, mu) | PC16 | |proj| | readout |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arm in ARMS:
            entry = result["correlations"]["within_arm"].get(arm)
            if not entry:
                continue
            cells = []
            for key in ("pr_w", "cos_mu", "pc16", "proj_abs_median", "readout"):
                value = entry.get(key, {}).get("rho")
                cells.append("–" if value is None else f"{value:.3f}")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines += ["", "## Across arms", "", "| layer | rho(PR(w), KL) |", "| --- | ---: |"]
        for L in layers:
            lines.append(f"| {L} | "
                         f"{result['correlations']['across_arms'][L]['rho_pr_kl']:.3f} |")
        r5 = result["bars"]["R5"]
        lines += ["", "## Gap", "",
                  f"OLS slope {r5['slope']:.3f}; raw n+ - full log-KL gap "
                  f"{r5['raw_gap']:.3f}; predicted from PR {r5['predicted_gap']:.3f}; "
                  f"explained {r5['explained']:.0%}.", ""]
    (out_dir / "reading-tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"gates {'PASS' if gates['passed'] else 'FAIL'}; "
          + (", ".join(f"{k}={bar_label(v).strip('*')}"
                       for k, v in result["bars"].items()) if gates["passed"] else "bars skipped"))
    print(f"wrote {out_dir/'reading.json'} and reading-tables.md in "
          f"{result['config']['seconds']}s")
    return 0 if gates["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
