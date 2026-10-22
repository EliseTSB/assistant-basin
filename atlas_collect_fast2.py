#!/usr/bin/env python3
"""Stream a Qwen3-1.7B neuron/control atlas from a local Pile JSONL sample.

The model is pinned to Part 49's Part 38 snapshot. Activations are reduced in
the down_proj pre-hook and never retained beyond one padded batch. CPU state is
bounded by top-20 candidates, maxima, threshold counts, and ten random samples
per unit. A second corpus pass is required because the 10% threshold depends on
the corpus-wide maximum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import part38


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/mnt/c/Users/alexa/neuron-probe-gpu/qwen35/pile10k.jsonl")
DEFAULT_OUT = ROOT / "results" / "atlas-qwen3-1.7b"
DEFAULT_RANDOM_OUT = ROOT / "results" / "atlas-qwen3-1.7b-random-init"
MODEL_ID = part38.MODEL_ID
MODEL_REVISION = part38.MODEL_REVISION
EXPECTED_LAYERS = 28
EXPECTED_WIDTH = 6144
MAX_DOC_TOKENS = 512
WINDOW_TOKENS = 64
FIRING_OFFSET = 32
TOP_K = 20
RANDOM_WINDOWS = 10
PASS_TWO_MAX_HITS = 20_000_000
DENSE_FALLBACK_FRACTION = 0.5


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(p) for p in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def parse_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        layers = sorted({int(piece.strip()) for piece in value.split(",") if piece.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("at least one layer is required")
    if any(layer < 0 or layer >= EXPECTED_LAYERS for layer in layers):
        raise argparse.ArgumentTypeError(f"layers must be between 0 and {EXPECTED_LAYERS - 1}")
    return layers


def generate_controls(n: int, width: int, seed: int, layer: int = 0) -> np.ndarray:
    """Return fixed-seed random unit-norm row directions."""
    if n < 0 or width <= 0:
        raise ValueError("control count must be nonnegative and width must be positive")
    rng = np.random.default_rng(stable_seed(seed, "control-directions", layer))
    values = rng.standard_normal((n, width), dtype=np.float32)
    if n:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        values /= np.maximum(norms, np.finfo(np.float32).tiny)
    return values


def merge_top20_torch(
    scores: Any,
    docs: Any,
    positions: Any,
    batch_scores: Any,
    batch_docs: Any,
    batch_positions: Any,
    k: int = TOP_K,
) -> tuple[Any, Any, Any]:
    """Merge [width,k] CPU state with per-document batch maxima using torch.topk."""
    import torch

    if scores.device.type != "cpu" or docs.device.type != "cpu" or positions.device.type != "cpu":
        raise ValueError("running top-k state must stay on CPU")
    if batch_scores.ndim != 2 or batch_positions.shape != batch_scores.shape:
        raise ValueError("batch scores and positions must have [docs,width] shape")
    if batch_scores.shape[1] != scores.shape[0] or batch_docs.numel() != batch_scores.shape[0]:
        raise ValueError("batch metadata does not match the top-k state width")
    batch_scores = batch_scores.detach().to(device="cpu", dtype=torch.float32)
    batch_positions = batch_positions.detach().to(device="cpu", dtype=torch.long)
    batch_docs = batch_docs.detach().to(device="cpu", dtype=torch.long)
    cand_scores = torch.cat((scores, batch_scores.T.contiguous()), dim=1)
    repeated_docs = batch_docs.reshape(1, -1).expand(scores.shape[0], -1)
    cand_docs = torch.cat((docs, repeated_docs), dim=1)
    cand_pos = torch.cat((positions, batch_positions.T.contiguous()), dim=1)
    keep = min(int(k), int(cand_scores.shape[1]))
    top_scores, top_idx = torch.topk(cand_scores, k=keep, dim=1, largest=True, sorted=True)
    top_docs = torch.gather(cand_docs, 1, top_idx)
    top_positions = torch.gather(cand_pos, 1, top_idx)
    if keep < k:
        pad = k - keep
        top_scores = torch.cat((top_scores, torch.full((scores.shape[0], pad), -math.inf)), dim=1)
        top_docs = torch.cat((top_docs, torch.full((scores.shape[0], pad), -1, dtype=torch.long)), dim=1)
        top_positions = torch.cat((top_positions, torch.full((scores.shape[0], pad), -1, dtype=torch.long)), dim=1)
    return top_scores, top_docs, top_positions


def decode_marked_window(
    tokenizer: Any,
    token_ids: Sequence[int],
    firing_pos: int,
    window_tokens: int = WINDOW_TOKENS,
    firing_offset: int = FIRING_OFFSET,
) -> str:
    """Decode a fixed-width token window and mark exactly the firing token.

    At a document boundary the absent context is represented by one explicit
    padding-count marker, keeping the firing token at the same token offset.
    """
    if not token_ids or not 0 <= firing_pos < len(token_ids):
        raise ValueError("firing position must identify a token in a nonempty document")
    if window_tokens <= 0 or not 0 <= firing_offset < window_tokens:
        raise ValueError("invalid window size or firing offset")
    start = firing_pos - firing_offset
    end = start + window_tokens
    real_start = max(0, start)
    real_end = min(len(token_ids), end)
    left_pad = max(0, -start)
    right_pad = max(0, end - len(token_ids))
    decode_kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    prefix = tokenizer.decode(list(token_ids[real_start:firing_pos]), **decode_kwargs)
    firing = tokenizer.decode([int(token_ids[firing_pos])], **decode_kwargs)
    suffix = tokenizer.decode(list(token_ids[firing_pos + 1:real_end]), **decode_kwargs)
    def escape_source(value: str) -> str:
        # Repeat until no adjacent pair is left: "[[[" -> "[ [[" still holds a pair after one pass (2026-10-22: the first full run
        # died here after two hours of collection).
        while "[[" in value or "]]" in value:
            value = value.replace("[[", "[ [").replace("]]", "] ]")
        return value
    prefix, firing, suffix = map(escape_source, (prefix, firing, suffix))
    if prefix.endswith("["):
        prefix += " "
    if firing.startswith("["):
        firing = " " + firing
    if firing.endswith("]"):
        firing += " "
    if suffix.startswith("]"):
        suffix = " " + suffix
    left = f"⟦PAD×{left_pad}⟧" if left_pad else ""
    right = f"⟦PAD×{right_pad}⟧" if right_pad else ""
    marked = f"{left}{prefix}[[{firing}]]{suffix}{right}"
    if marked.count("[[") != 1 or marked.count("]]") != 1:
        # Last resort: drop every source bracket so exactly the firing-token pair remains (a window is never allowed to kill a run).
        strip = lambda value: value.replace("[", "(").replace("]", ")")
        marked = f"{left}{strip(prefix)}[[{strip(firing)}]]{strip(suffix)}{right}"
    assert marked.count("[[") == 1 and marked.count("]]") == 1
    return marked


def _read_docs(path: Path, tokenizer: Any, limit: int | None) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    texts: list[str] = []
    source_ids: list[Any] = []
    with path.open(encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream):
            if limit is not None and ordinal >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"JSONL row {ordinal} must be an object with a text string")
            texts.append(row["text"])
            source_ids.append(row.get("i", ordinal))
            # Tokenize in moderate chunks below to avoid a large temporary batch.
            if len(texts) == 128:
                _append_tokenized_docs(docs, texts, source_ids, tokenizer)
                texts, source_ids = [], []
    if texts:
        _append_tokenized_docs(docs, texts, source_ids, tokenizer)
    if not docs:
        raise ValueError(f"no documents were read from {path}")
    return docs


def _append_tokenized_docs(
    out: list[dict[str, Any]], texts: list[str], source_ids: list[Any], tokenizer: Any
) -> None:
    encoded = tokenizer(texts, truncation=True, max_length=MAX_DOC_TOKENS,
                        padding=False, add_special_tokens=True)
    id_rows = encoded["input_ids"]
    for text, source_id, values in zip(texts, source_ids, id_rows):
        ids = [int(value) for value in values]
        if not ids:
            continue
        out.append({"ordinal": len(out), "doc": source_id, "ids": ids})


def make_batches(docs: Sequence[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    """Length-sort documents so padding waste is low and batch boundaries repeat."""
    ordered = sorted(docs, key=lambda row: (len(row["ids"]), int(row["ordinal"])))
    return [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]


class WindowReservoir:
    """Keep the k positions with the smallest seeded random priorities per unit."""

    def __init__(self, width: int, count: int, seed: int, layer: int, kind: str):
        self.width = width
        self.count = count
        self.seed = stable_seed(seed, "window-reservoir", kind, layer)
        self.keys = np.full((width, count), np.inf, dtype=np.float32)
        self.docs = np.full((width, count), -1, dtype=np.int64)
        self.positions = np.full((width, count), -1, dtype=np.int64)
        self.acts = np.full((width, count), np.nan, dtype=np.float32)
        self.rng = np.random.default_rng(self.seed)

    def _merge_unit(self, unit: int, batch_keys: np.ndarray, batch_docs: np.ndarray,
                    batch_positions: np.ndarray, batch_acts: np.ndarray) -> None:
        if not self.count or not len(batch_keys):
            return
        candidate_keys = np.concatenate((self.keys[unit], batch_keys))
        candidate_docs = np.concatenate((self.docs[unit], batch_docs))
        candidate_positions = np.concatenate((self.positions[unit], batch_positions))
        candidate_acts = np.concatenate((self.acts[unit], batch_acts))
        pick = np.argpartition(candidate_keys, self.count - 1)[:self.count]
        picked_keys = candidate_keys[pick]
        finite = np.isfinite(picked_keys)
        self.keys[unit] = np.where(finite, picked_keys, np.inf)
        self.docs[unit] = np.where(finite, candidate_docs[pick], -1)
        self.positions[unit] = np.where(finite, candidate_positions[pick], -1)
        self.acts[unit] = np.where(finite, candidate_acts[pick], np.nan)

    def update_sparse(self, units: np.ndarray, flat_tokens: np.ndarray,
                      values: np.ndarray, docs: Sequence[dict[str, Any]],
                      token_count: int) -> None:
        """Merge only qualifying (unit, flat-token, value) candidates."""
        if not self.count or not len(units):
            return
        units = np.asarray(units, dtype=np.int64)
        flat_tokens = np.asarray(flat_tokens, dtype=np.int64)
        values = np.asarray(values, dtype=np.float32)
        batch_ordinals = np.asarray([int(row["ordinal"]) for row in docs], dtype=np.int64)
        batch_docs = batch_ordinals[flat_tokens // token_count]
        batch_positions = flat_tokens % token_count
        batch_keys = self.rng.random(len(units), dtype=np.float32)

        order = np.argsort(units, kind="stable")
        sorted_units = units[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_units)) + 1]
        ends = np.r_[starts[1:], len(order)]
        for start, end in zip(starts, ends):
            unit = int(sorted_units[start])
            pick = order[start:end]
            self._merge_unit(unit, batch_keys[pick], batch_docs[pick],
                             batch_positions[pick], values[pick])

    def update_dense_unit(self, unit: int, values: np.ndarray, qualifies: np.ndarray,
                          docs: Sequence[dict[str, Any]], token_count: int,
                          token_start: int) -> None:
        """Use the dense NumPy update for a row whose qualifying mask is dense."""
        if not self.count:
            return
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        qualifies = np.asarray(qualifies, dtype=bool).reshape(-1)
        batch_keys = self.rng.random(len(values), dtype=np.float32)
        batch_keys[~qualifies] = np.inf
        flat_tokens = np.arange(token_start, token_start + len(values), dtype=np.int64)
        batch_ordinals = np.asarray([int(row["ordinal"]) for row in docs], dtype=np.int64)
        batch_docs = batch_ordinals[flat_tokens // token_count]
        batch_positions = flat_tokens % token_count
        self._merge_unit(int(unit), batch_keys, batch_docs, batch_positions, values)


class UnitState:
    def __init__(self, width: int, count: int, seed: int, layer: int, kind: str,
                 input_width: int | None = None, unit_ids: Sequence[int] | None = None):
        import torch

        self.width = width
        self.count = count
        self.kind = kind
        self.unit_ids = np.asarray(list(unit_ids) if unit_ids is not None else range(width), dtype=np.int64)
        if len(self.unit_ids) != width:
            raise ValueError("unit_ids must have one entry for every output coordinate")
        self.max_values = np.full(width, -math.inf, dtype=np.float32)
        self.top_scores = torch.full((width, TOP_K), -math.inf, dtype=torch.float32, device="cpu")
        self.top_docs = torch.full((width, TOP_K), -1, dtype=torch.long, device="cpu")
        self.top_positions = torch.full((width, TOP_K), -1, dtype=torch.long, device="cpu")
        self.above_count = np.zeros(width, dtype=np.int64)
        self.random = WindowReservoir(width, RANDOM_WINDOWS, seed, layer, f"{kind}-random")
        self.weak = WindowReservoir(width, 5, seed, layer, f"{kind}-weak")
        self.directions: Any | None = None
        if kind == "control":
            self.directions = generate_controls(count, int(input_width or width), seed, layer)

    def finish_second_pass(self) -> None:
        # Maxima, counts, and reservoir rows stay in the same CPU layout consumed
        # by _record_rows throughout pass two.
        return None

    def record_first_pass(self, acts: Any, docs: Sequence[dict[str, Any]], mask: Any) -> None:
        import torch

        masked = acts.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        doc_values, doc_pos = masked.max(dim=1)
        if not torch.isfinite(doc_values).all():
            raise RuntimeError("a document batch had no valid token positions")
        self.max_values = np.maximum(
            self.max_values,
            doc_values.max(dim=0).values.detach().float().cpu().numpy(),
        )
        doc_ordinals = torch.tensor([row["ordinal"] for row in docs], dtype=torch.long)
        self.top_scores, self.top_docs, self.top_positions = merge_top20_torch(
            self.top_scores, self.top_docs, self.top_positions,
            doc_values.detach().float().cpu(), doc_ordinals, doc_pos.detach().cpu(), TOP_K)

    def record_second_pass(self, acts: Any, docs: Sequence[dict[str, Any]], mask: Any) -> None:
        import torch

        if not self.width:
            return
        flat_acts = acts.detach().reshape(-1, self.width)
        flat_valid = mask.to(device=acts.device, dtype=torch.bool).reshape(-1)
        max_values = torch.as_tensor(self.max_values, dtype=torch.float32, device=acts.device)
        token_count = int(acts.shape[1])
        rows_per_chunk = max(1, PASS_TWO_MAX_HITS // self.width)

        def update_reservoir(reservoir: WindowReservoir, values: Any, qualifies: Any,
                             hit_counts: Any, valid_count: Any, token_start: int) -> None:
            minimum_dense_hits = torch.ceil(
                valid_count.to(dtype=torch.float32) * DENSE_FALLBACK_FRACTION
            ).to(dtype=torch.long)
            dense = (hit_counts >= minimum_dense_hits) & (valid_count > 0)
            dense_units = torch.nonzero(dense, as_tuple=False).flatten().cpu().tolist()
            for unit in dense_units:
                reservoir.update_dense_unit(
                    int(unit),
                    values[:, int(unit)].detach().cpu().numpy(),
                    qualifies[:, int(unit)].detach().cpu().numpy(),
                    docs, token_count, token_start,
                )

            sparse_qualifies = qualifies & ~dense.unsqueeze(0)
            pair_indices = torch.nonzero(
                sparse_qualifies.reshape(-1), as_tuple=False
            ).flatten()
            if not pair_indices.numel():
                return
            pair_values = values.reshape(-1).index_select(0, pair_indices)
            pair_indices_cpu = pair_indices.detach().cpu().numpy()
            pair_values_cpu = pair_values.detach().cpu().numpy()
            reservoir.update_sparse(
                pair_indices_cpu % self.width,
                pair_indices_cpu // self.width + token_start,
                pair_values_cpu,
                docs,
                token_count,
            )

        for token_start in range(0, int(flat_acts.shape[0]), rows_per_chunk):
            token_stop = min(token_start + rows_per_chunk, int(flat_acts.shape[0]))
            values = flat_acts[token_start:token_stop].float()
            valid = flat_valid[token_start:token_stop]
            valid_count = valid.sum(dtype=torch.long)
            ten_pct = max_values * 0.1
            qualifies = (values > ten_pct.unsqueeze(0)) & valid.unsqueeze(1)
            hit_counts = qualifies.sum(dim=0, dtype=torch.long)
            self.above_count += hit_counts.detach().cpu().numpy()
            update_reservoir(self.random, values, qualifies, hit_counts, valid_count, token_start)

            thirty_pct = max_values * 0.3
            lower = torch.minimum(ten_pct, thirty_pct)
            upper = torch.maximum(ten_pct, thirty_pct)
            weak_qualifies = ((values > lower.unsqueeze(0))
                              & (values <= upper.unsqueeze(0)) & valid.unsqueeze(1))
            weak_counts = weak_qualifies.sum(dim=0, dtype=torch.long)
            update_reservoir(self.weak, values, weak_qualifies, weak_counts,
                             valid_count, token_start)


def _make_state(layer: int, n_controls: int, seed: int,
                neuron_units: Sequence[int] | None = None) -> tuple[UnitState, UnitState]:
    unit_ids = list(range(EXPECTED_WIDTH)) if neuron_units is None else list(neuron_units)
    return (
        UnitState(len(unit_ids), len(unit_ids), seed, layer, "neuron", unit_ids=unit_ids),
        UnitState(n_controls, n_controls, seed, layer, "control", input_width=EXPECTED_WIDTH),
    )


def sample_random_units(layers: Sequence[int], count: int, seed: int,
                        width: int = EXPECTED_WIDTH) -> dict[int, list[int]]:
    """Choose a reproducible sample of (layer, unit) coordinates."""
    candidates = [(int(layer), unit) for layer in layers for unit in range(width)]
    if count <= 0 or count > len(candidates):
        raise ValueError(f"random-init sample count must be in [1, {len(candidates)}]")
    chosen = random.Random(seed).sample(candidates, count)
    selected: dict[int, list[int]] = {int(layer): [] for layer in layers}
    for layer, unit in chosen:
        selected[layer].append(unit)
    return {layer: sorted(units) for layer, units in selected.items() if units}


def select_random_init_units(layers: Sequence[int], count: int, seed: int) -> dict[int, list[int]]:
    """Select one fixed sample across all layers, then filter to this invocation's layer slice."""
    all_units = sample_random_units(list(range(EXPECTED_LAYERS)), count, seed)
    requested = {int(layer) for layer in layers}
    return {layer: units for layer, units in all_units.items() if layer in requested}


def load_random_init_model(args: argparse.Namespace):
    """Construct the pinned architecture from local config without loading weights."""
    torch, AutoModelForCausalLM, AutoTokenizer = part38._torch_runtime()
    from transformers import AutoConfig

    snapshot = part38.resolve_snapshot(args.cache_dir, MODEL_ID, MODEL_REVISION)
    if args.device == "xpu" and not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
        raise RuntimeError("--device xpu requested but no XPU is available")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    dtype = torch.bfloat16 if args.device in ("xpu", "cuda") else torch.float32
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(args.random_init_seed))
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    model.to(args.device).eval()
    torch.set_grad_enabled(False)
    layers = part38.find_text_layers(model, torch)
    return torch, tokenizer, model, layers, snapshot


class _StopAfterLastHook(Exception):
    """Private control-flow exception used to skip unrequested upper layers."""


def _forward_batches(
    torch: Any,
    base_model: Any,
    layers: Any,
    selected_states: dict[int, tuple[UnitState, UnitState]],
    batches: Sequence[Sequence[dict[str, Any]]],
    device: str,
    pad_id: int,
    pass_number: int,
) -> None:
    handles = []
    current: dict[str, Any] = {}
    directions_device: dict[int, Any] = {}
    highest_layer = max(selected_states)

    for layer_idx, states in selected_states.items():
        neuron_state, control_state = states
        down_proj = layers[layer_idx].mlp.down_proj
        width = int(down_proj.in_features)
        if width != EXPECTED_WIDTH:
            raise ValueError(f"layer {layer_idx} width {width} differs from expected {EXPECTED_WIDTH}")
        if control_state.count:
            directions_device[layer_idx] = torch.as_tensor(
                control_state.directions, dtype=torch.float32, device=device)

        def hook(_module: Any, args: tuple[Any, ...], *, index: int = layer_idx,
                 real_state: UnitState = neuron_state, ctrl_state: UnitState = control_state):
            x = args[0]
            if x.ndim != 3 or x.shape[-1] != EXPECTED_WIDTH:
                raise ValueError(f"layer {index} hook saw unexpected shape {tuple(x.shape)}")
            mask = current["mask"]
            if real_state.width == EXPECTED_WIDTH and np.array_equal(
                    real_state.unit_ids, np.arange(EXPECTED_WIDTH, dtype=np.int64)):
                real_state_acts = x.detach()
            else:
                unit_index = torch.as_tensor(real_state.unit_ids, dtype=torch.long, device=x.device)
                real_state_acts = x.detach().index_select(-1, unit_index)
            ctrl_acts = None
            if ctrl_state.count:
                ctrl_acts = torch.matmul(
                    x.detach().float(), directions_device[index].transpose(0, 1))
            if pass_number == 1:
                real_state.record_first_pass(real_state_acts, current["docs"], mask)
                if ctrl_acts is not None:
                    ctrl_state.record_first_pass(ctrl_acts, current["docs"], mask)
            else:
                real_state.record_second_pass(real_state_acts, current["docs"], mask)
                if ctrl_acts is not None:
                    ctrl_state.record_second_pass(ctrl_acts, current["docs"], mask)
            return None

        handles.append(down_proj.register_forward_pre_hook(hook))
        if layer_idx == highest_layer:
            def stop_after_last_layer(_module: Any, _args: tuple[Any, ...], _output: Any):
                raise _StopAfterLastHook

            handles.append(down_proj.register_forward_hook(stop_after_last_layer))

    try:
        with torch.inference_mode():
            for batch in batches:
                max_len = max(len(row["ids"]) for row in batch)
                ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
                mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
                for row_idx, row in enumerate(batch):
                    row_ids = torch.as_tensor(row["ids"], dtype=torch.long)
                    ids[row_idx, :len(row["ids"])] = row_ids
                    mask[row_idx, :len(row["ids"])] = True
                current["docs"] = batch
                current["mask"] = mask.to(device)
                try:
                    base_model(input_ids=ids.to(device), attention_mask=current["mask"],
                               use_cache=False, return_dict=True)
                except _StopAfterLastHook:
                    pass
    finally:
        for handle in handles:
            handle.remove()
    if pass_number == 2:
        for neuron_state, control_state in selected_states.values():
            neuron_state.finish_second_pass()
            control_state.finish_second_pass()


def _record_rows(
    layer: int,
    state: UnitState,
    docs: Sequence[dict[str, Any]],
    tokenizer: Any,
    total_positions: int,
    model_arm: str = "trained",
) -> list[dict[str, Any]]:
    id_to_doc = {int(row["ordinal"]): row for row in docs}
    rows: list[dict[str, Any]] = []
    for unit_idx in range(state.width):
        if state.kind == "control" and unit_idx >= state.count:
            break
        top: list[dict[str, Any]] = []
        for score, doc_ordinal, pos in zip(
            state.top_scores[unit_idx].tolist(),
            state.top_docs[unit_idx].tolist(),
            state.top_positions[unit_idx].tolist(),
        ):
            if doc_ordinal < 0 or not math.isfinite(score):
                continue
            doc = id_to_doc[int(doc_ordinal)]
            top.append({
                "doc": doc["doc"], "pos": int(pos), "act": float(score),
                "window": decode_marked_window(tokenizer, doc["ids"], int(pos)),
            })
        def materialize(reservoir: WindowReservoir) -> list[dict[str, Any]]:
            selected: list[dict[str, Any]] = []
            for doc_ordinal, pos, act in zip(
                reservoir.docs[unit_idx].tolist(),
                reservoir.positions[unit_idx].tolist(),
                reservoir.acts[unit_idx].tolist(),
            ):
                if doc_ordinal < 0 or pos < 0 or not math.isfinite(act):
                    continue
                doc = id_to_doc[int(doc_ordinal)]
                selected.append({
                    "doc": doc["doc"], "pos": int(pos), "act": float(act),
                    "window": decode_marked_window(tokenizer, doc["ids"], int(pos)),
                })
            return selected

        random_rows = materialize(state.random)
        weak_rows = materialize(state.weak)
        record: dict[str, Any] = {
            "layer": int(layer), "max": float(state.max_values[unit_idx]),
            "frac_above": float(state.above_count[unit_idx] / total_positions),
            "top": top, "random": random_rows, "weak": weak_rows,
            "model_arm": model_arm,
        }
        if state.kind == "neuron":
            record["unit"] = int(state.unit_ids[unit_idx])
        else:
            record["control_id"] = int(unit_idx)
        rows.append(record)
    return rows


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def layers_to_run(out_dir: Path, layers: Sequence[int], config: dict[str, Any],
                  selected_units: dict[int, Sequence[int]] | None = None) -> list[int]:
    """Return requested layers lacking matching, complete per-layer checkpoints."""
    checkpoint_dir = out_dir / "checkpoints"
    todo: list[int] = []
    for layer in layers:
        base = f"layer-{int(layer):02d}"
        done_path = checkpoint_dir / f"{base}.done.json"
        neuron_path = checkpoint_dir / f"neurons-{base}.jsonl"
        control_path = checkpoint_dir / f"controls-{base}.jsonl"
        try:
            meta = json.loads(done_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            todo.append(int(layer))
            continue
        selection_matches = (selected_units is None or
                             meta.get("selected_unit_sha256") == _unit_selection_sha256(selected_units[int(layer)]))
        if (meta.get("config") != config or not selection_matches or
                not neuron_path.is_file() or not control_path.is_file()):
            todo.append(int(layer))
    return todo


def _config(args: argparse.Namespace, selected_units: dict[int, Sequence[int]]) -> dict[str, Any]:
    data = args.data.resolve()
    stat = data.stat()
    return {
        "data": str(data), "data_size": stat.st_size, "data_mtime_ns": stat.st_mtime_ns,
        "data_edge_sha256": _corpus_edge_sha256(data),
        "docs_limit": args.docs, "model": MODEL_ID, "revision": MODEL_REVISION,
        "seed": args.seed, "n_controls": args.n_controls, "batch_size": args.batch_size,
        "max_doc_tokens": MAX_DOC_TOKENS, "window_tokens": WINDOW_TOKENS,
        "firing_offset": FIRING_OFFSET, "top_k": TOP_K, "random_windows": RANDOM_WINDOWS,
        "device": args.device, "dtype": "bf16" if args.device in ("xpu", "cuda") else "fp32",
        "model_arm": "random-init" if args.random_init else "trained",
        "random_init_seed": args.random_init_seed if args.random_init else None,
        "random_init_units": args.random_init_units if args.random_init else None,
    }


def _corpus_edge_sha256(path: Path) -> str:
    """Hash the first and last MiB so same-size, restored-mtime edits are detected."""
    edge_size = 1024 * 1024
    with path.open("rb") as stream:
        first = stream.read(edge_size)
        stream.seek(max(0, path.stat().st_size - edge_size))
        last = stream.read(edge_size)
    digest = hashlib.sha256()
    digest.update(len(first).to_bytes(8, "big"))
    digest.update(first)
    digest.update(len(last).to_bytes(8, "big"))
    digest.update(last)
    return digest.hexdigest()


def _unit_selection_sha256(units: Sequence[int]) -> str:
    payload = json.dumps(list(units), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_markers(checkpoint_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    markers: list[tuple[Path, dict[str, Any]]] = []
    for pattern in ("layer-*.done.json", "layer-*.pass1.json"):
        for marker in sorted(checkpoint_dir.glob(pattern)):
            try:
                meta = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(meta, dict) and isinstance(meta.get("config"), dict):
                markers.append((marker, meta))
    return markers


def _assert_compatible_checkpoint_configs(out_dir: Path, config: dict[str, Any],
                                          selected_units: dict[int, Sequence[int]] | None = None) -> None:
    markers = _checkpoint_markers(out_dir / "checkpoints")
    current_hash = _config_sha256(config)
    if not markers and any((out_dir / f"{stem}.jsonl").exists() for stem in ("neurons", "controls")):
        raise ValueError(
            f"checkpoint config hashes differ: existing=unavailable (legacy merged output), "
            f"current={current_hash}; choose a new --out directory"
        )
    for marker, meta in markers:
        existing_hash = _config_sha256(meta["config"])
        if existing_hash != current_hash:
            raise ValueError(
                f"checkpoint config hashes differ: existing={existing_hash} ({marker.name}), "
                f"current={current_hash}; choose a new --out directory or restore the original config"
            )
        if selected_units is not None:
            layer = meta.get("layer")
            if layer is None:
                try:
                    layer = int(marker.name.split("-")[1].split(".")[0])
                except (ValueError, IndexError):
                    continue
            layer = int(layer)
            if layer in selected_units:
                expected_selection = _unit_selection_sha256(selected_units[layer])
                existing_selection = meta.get("selected_unit_sha256")
                merged_exists = any((out_dir / f"{stem}.jsonl").is_file()
                                    for stem in ("neurons", "controls"))
                if existing_selection is None and merged_exists:
                    raise ValueError(
                        f"checkpoint selected-unit hash is unavailable for merged layer {layer}; "
                        "choose a new --out"
                    )
                if existing_selection and existing_selection != expected_selection:
                    raise ValueError(
                        f"checkpoint selected-unit hashes differ for layer {layer}: "
                        f"existing={existing_selection}, current={expected_selection}; choose a new --out"
                    )


def _jsonl_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for _ in stream)


def _config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def layers_with_pass_one_checkpoint(out_dir: Path, layers: Sequence[int], config: dict[str, Any],
                                    selected_units: dict[int, Sequence[int]]) -> list[int]:
    checkpoint_dir = out_dir / "checkpoints"
    reusable: list[int] = []
    for layer in layers:
        marker = checkpoint_dir / f"layer-{int(layer):02d}.pass1.json"
        state_path = checkpoint_dir / f"layer-{int(layer):02d}.pass1.npz"
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (meta.get("config") == config and
                meta.get("selected_unit_sha256") == _unit_selection_sha256(selected_units[int(layer)]) and
                meta.get("status") == "pass1_complete" and state_path.is_file()):
            reusable.append(int(layer))
    return reusable


def save_pass_one_checkpoint(out_dir: Path, layer: int, config: dict[str, Any],
                             neuron_state: UnitState, control_state: UnitState,
                             selected_units: Sequence[int] | None = None) -> None:
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base = f"layer-{int(layer):02d}.pass1"
    state_path = checkpoint_dir / f"{base}.npz"
    temp = state_path.with_name(state_path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    with temp.open("wb") as stream:
        np.savez(
            stream,
            neuron_max=neuron_state.max_values,
            neuron_top_scores=neuron_state.top_scores.numpy(),
            neuron_top_docs=neuron_state.top_docs.numpy(),
            neuron_top_positions=neuron_state.top_positions.numpy(),
            control_max=control_state.max_values,
            control_top_scores=control_state.top_scores.numpy(),
            control_top_docs=control_state.top_docs.numpy(),
            control_top_positions=control_state.top_positions.numpy(),
        )
    temp.replace(state_path)
    units = selected_units if selected_units is not None else neuron_state.unit_ids.tolist()
    _write_json_atomic(checkpoint_dir / f"{base}.json", {
        "layer": int(layer), "config": config, "selected_unit_sha256": _unit_selection_sha256(units),
        "status": "pass1_complete", "state_file": state_path.name,
    })


def load_pass_one_checkpoint(out_dir: Path, layer: int, config: dict[str, Any],
                             seed: int, n_controls: int,
                             neuron_units: Sequence[int]) -> tuple[UnitState, UnitState]:
    import torch

    checkpoint_dir = out_dir / "checkpoints"
    marker = checkpoint_dir / f"layer-{int(layer):02d}.pass1.json"
    state_path = checkpoint_dir / f"layer-{int(layer):02d}.pass1.npz"
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta.get("config") != config or meta.get("selected_unit_sha256") != _unit_selection_sha256(neuron_units):
        raise ValueError(f"pass-one checkpoint does not match current config for layer {layer}")
    neuron_state, control_state = _make_state(layer, n_controls, seed, neuron_units)
    with np.load(state_path) as saved:
        neuron_state.max_values[:] = saved["neuron_max"]
        neuron_state.top_scores[:] = torch.from_numpy(saved["neuron_top_scores"])
        neuron_state.top_docs[:] = torch.from_numpy(saved["neuron_top_docs"])
        neuron_state.top_positions[:] = torch.from_numpy(saved["neuron_top_positions"])
        control_state.max_values[:] = saved["control_max"]
        control_state.top_scores[:] = torch.from_numpy(saved["control_top_scores"])
        control_state.top_docs[:] = torch.from_numpy(saved["control_top_docs"])
        control_state.top_positions[:] = torch.from_numpy(saved["control_top_positions"])
    return neuron_state, control_state


def _merge_checkpoints(out_dir: Path, config: dict[str, Any] | None = None) -> None:
    checkpoint_dir = out_dir / "checkpoints"
    marker_records = _checkpoint_markers(checkpoint_dir)
    merge_manifest_path = out_dir / "merge-manifest.json"
    existing_merge: dict[str, Any] = {}
    if merge_manifest_path.is_file():
        try:
            existing_merge = json.loads(merge_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ValueError("merge-manifest.json is unreadable; refusing to replace merged atlas")
    if config is not None:
        current_hash = _config_sha256(config)
        for marker, meta in marker_records:
            existing_hash = _config_sha256(meta["config"])
            if existing_hash != current_hash:
                raise ValueError(
                    f"checkpoint config hashes differ: existing={existing_hash} ({marker.name}), "
                    f"current={current_hash}; merged files were not changed"
                )
    elif marker_records:
        known_hashes = {_config_sha256(meta["config"]) for _, meta in marker_records}
        if len(known_hashes) > 1:
            hashes = ", ".join(sorted(known_hashes))
            raise ValueError(f"checkpoint config hashes differ: {hashes}; merged files were not changed")
    if not marker_records and any((out_dir / f"{stem}.jsonl").exists() for stem in ("neurons", "controls")):
        raise ValueError("merged atlas exists without checkpoint config hashes; refusing to shrink it")
    complete: list[int] = []
    for marker in sorted(checkpoint_dir.glob("layer-*.done.json")):
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
            layer = int(meta["layer"])
            if (checkpoint_dir / f"neurons-layer-{layer:02d}.jsonl").is_file() and (
                    checkpoint_dir / f"controls-layer-{layer:02d}.jsonl").is_file():
                complete.append(layer)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    selected_by_layer: dict[str, str | None] = {}
    for layer in complete:
        done_path = checkpoint_dir / f"layer-{layer:02d}.done.json"
        meta = json.loads(done_path.read_text(encoding="utf-8"))
        selected_by_layer[str(layer)] = meta.get("selected_unit_sha256")
    old_selected = existing_merge.get("layers", {})
    merged_exists = any((out_dir / f"{stem}.jsonl").is_file() for stem in ("neurons", "controls"))
    if merged_exists and isinstance(old_selected, dict):
        for layer, old_sha in old_selected.items():
            if layer in selected_by_layer and selected_by_layer[layer] != old_sha:
                raise ValueError(
                    f"refusing to replace merged layer {layer} with a different selected-unit hash"
                )
    for stem in ("neurons", "controls"):
        target = out_dir / f"{stem}.jsonl"
        if target.is_file():
            existing_count = _jsonl_row_count(target)
            candidate_count = sum(_jsonl_row_count(
                checkpoint_dir / f"{stem}-layer-{layer:02d}.jsonl") for layer in complete)
            if candidate_count < existing_count:
                raise ValueError(
                    f"refusing to shrink {target.name}: existing={existing_count} rows, "
                    f"complete checkpoints={candidate_count} rows"
                )
    # Checkpoints are already ordered by layer and unit. Concatenate them as
    # streams so a large atlas is never materialized as Python records in RAM.
    for stem in ("neurons", "controls"):
        target = out_dir / f"{stem}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + f".{os.getpid()}.{time.time_ns()}.tmp")
        with temp.open("w", encoding="utf-8") as destination:
            for layer in complete:
                checkpoint = checkpoint_dir / f"{stem}-layer-{layer:02d}.jsonl"
                with checkpoint.open(encoding="utf-8") as source:
                    for line in source:
                        destination.write(line)
        temp.replace(target)
    _write_json_atomic(merge_manifest_path, {
        "config_sha256": (_config_sha256(config) if config is not None else None),
        "layers": selected_by_layer,
        "merged_at": time.time(),
    })


def estimate_host_ram_peak_bytes(layer_count: int, n_controls: int, batch_size: int,
                                neuron_units_per_layer: int = EXPECTED_WIDTH) -> int:
    """Estimate state plus peak CPU pass-two workspace, excluding model/doc storage.

    The per-unit persistent state includes maxima, top-20, and both reservoirs.
    The transient multiplier conservatively accounts for values, both qualification
    masks, reservoir priorities, candidate keys, and argpartition indices.
    """
    directions = int(layer_count) * int(n_controls) * EXPECTED_WIDTH * 4
    state_units = int(layer_count) * (int(n_controls) + int(neuron_units_per_layer))
    bytes_per_unit_state = 4 + TOP_K * (4 + 8 + 8) + 8 + (RANDOM_WINDOWS + 5) * (4 + 8 + 8 + 4)
    persistent = state_units * bytes_per_unit_state
    transient = int(batch_size) * MAX_DOC_TOKENS * int(n_controls) * 22
    return directions + persistent + transient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Pile sample JSONL (each row has a text field)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=part38.DEFAULT_CACHE)
    parser.add_argument("--layers", type=parse_layers, default=None,
                        help=("comma-separated layers; default is all 28. Pass-one checkpoints are written "
                              "after the shared pass finishes for every layer in this call; use 7-layer calls "
                              "(0-6, 7-13, 14-20, 21-27) to limit restart cost"))
    parser.add_argument("--docs", type=int, default=None,
                        help="limit to the first N JSONL rows for a pilot")
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-controls", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=5601)
    parser.add_argument("--random-init", action="store_true",
                        help="construct the pinned Qwen3 architecture from config only")
    parser.add_argument("--random-init-seed", type=int, default=5602)
    parser.add_argument("--random-init-units", type=int, default=2000,
                        help="fixed-seed neuron sample size when --random-init is selected")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = DEFAULT_RANDOM_OUT if args.random_init else DEFAULT_OUT
    if args.docs is not None and args.docs <= 0:
        raise SystemExit("--docs must be positive")
    if args.batch_size <= 0 or args.n_controls < 0:
        raise SystemExit("--batch-size must be positive and --n-controls nonnegative")
    if args.random_init_units <= 0:
        raise SystemExit("--random-init-units must be positive")
    if not args.data.is_file():
        raise SystemExit(f"Pile JSONL not found: {args.data}")
    requested_layers = args.layers or list(range(EXPECTED_LAYERS))
    selected_units = (select_random_init_units(requested_layers, args.random_init_units, args.random_init_seed)
                      if args.random_init else {layer: list(range(EXPECTED_WIDTH)) for layer in requested_layers})
    if not selected_units:
        raise SystemExit("no sampled random-init units fall in the requested --layers slice")
    effective_layers = sorted(selected_units)
    config = _config(args, selected_units)
    expected_direction_bytes = len(effective_layers) * args.n_controls * EXPECTED_WIDTH * 4
    full_direction_bytes = EXPECTED_LAYERS * args.n_controls * EXPECTED_WIDTH * 4
    neuron_units_per_layer = max((len(units) for units in selected_units.values()), default=0)
    expected_peak_bytes = estimate_host_ram_peak_bytes(
        len(effective_layers), args.n_controls, args.batch_size, neuron_units_per_layer)
    transient_bytes = args.batch_size * MAX_DOC_TOKENS * args.n_controls * 22
    print(f"estimated host working-set peak (directions + state + pass-two batch workspace): "
          f"{expected_peak_bytes / 1e9:.2f} GB this call; {expected_direction_bytes / 1e9:.2f} GB "
          f"control directions; {transient_bytes / 1e6:.1f} MB per-batch workspace; "
          f"{full_direction_bytes / 1e9:.2f} GB directions for all {EXPECTED_LAYERS} layers; "
          "the 1.2 GB admission cap applies to directions only; use 7-layer calls for the full run", flush=True)
    if expected_direction_bytes > 1_200_000_000:
        raise SystemExit("control directions exceed the 1.2 GB per-call direction budget; "
                         "run at most 7 layers per call or lower --n-controls")
    _assert_compatible_checkpoint_configs(args.out, config, selected_units)
    todo = layers_to_run(args.out, effective_layers, config, selected_units)
    if not todo:
        _merge_checkpoints(args.out, config)
        print(f"all requested layers already complete: {effective_layers}")
        return 0

    load_args = argparse.Namespace(
        cache_dir=str(args.cache_dir), model=MODEL_ID, revision=MODEL_REVISION,
        device=args.device, dtype="bf16" if args.device in ("xpu", "cuda") else "fp32", threads=0,
        random_init_seed=args.random_init_seed,
    )
    if args.random_init:
        torch, tokenizer, model, layers, snapshot = load_random_init_model(load_args)
    else:
        torch, tokenizer, model, layers, snapshot = part38.load_model(load_args)
    if len(layers) != EXPECTED_LAYERS:
        raise RuntimeError(f"pinned model exposed {len(layers)} layers, expected {EXPECTED_LAYERS}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("pinned causal LM has no base .model module")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    docs = _read_docs(args.data, tokenizer, args.docs)
    batches = make_batches(docs, args.batch_size)
    total_positions = sum(len(row["ids"]) for row in docs)
    if total_positions <= 0:
        raise RuntimeError("no valid token positions in the selected documents")
    reusable_pass_one = set(layers_with_pass_one_checkpoint(args.out, todo, config, selected_units))
    states = {
        layer: (load_pass_one_checkpoint(args.out, layer, config, args.seed, args.n_controls,
                                         selected_units[layer]) if layer in reusable_pass_one else
                _make_state(layer, args.n_controls, args.seed, selected_units[layer]))
        for layer in todo
    }
    arm = "random-init" if args.random_init else "trained"
    print(f"model {MODEL_ID}@{MODEL_REVISION} from config/snapshot {snapshot}; arm={arm}; "
          f"{len(docs)} docs, {total_positions} tokens; layers {todo}; device={args.device}", flush=True)
    pass_one_states = {layer: states[layer] for layer in todo if layer not in reusable_pass_one}
    if pass_one_states:
        _forward_batches(torch, base_model, layers, pass_one_states, batches, args.device,
                         int(tokenizer.pad_token_id), pass_number=1)
        for layer in pass_one_states:
            neuron_state, control_state = states[layer]
            save_pass_one_checkpoint(args.out, layer, config, neuron_state, control_state,
                                     selected_units[layer])
    if reusable_pass_one:
        print(f"resuming pass two from saved pass-one state for layers {sorted(reusable_pass_one)}", flush=True)
    _forward_batches(torch, base_model, layers, states, batches, args.device,
                     int(tokenizer.pad_token_id), pass_number=2)

    checkpoint_dir = args.out / "checkpoints"
    for layer in todo:
        neuron_state, control_state = states[layer]
        neuron_rows = _record_rows(layer, neuron_state, docs, tokenizer, total_positions, arm)
        control_rows = _record_rows(layer, control_state, docs, tokenizer, total_positions, arm)
        base = f"layer-{layer:02d}"
        _write_jsonl_atomic(checkpoint_dir / f"neurons-{base}.jsonl", neuron_rows)
        _write_jsonl_atomic(checkpoint_dir / f"controls-{base}.jsonl", control_rows)
        _write_json_atomic(checkpoint_dir / f"{base}.done.json", {
            "layer": layer, "config": config,
            "selected_unit_sha256": _unit_selection_sha256(selected_units[layer]),
            "docs_processed": len(docs),
            "positions": total_positions, "completed_unix": time.time(),
        })
        print(f"saved layer {layer}: {len(neuron_rows)} neurons, {len(control_rows)} controls", flush=True)
    _merge_checkpoints(args.out, config)
    print(f"wrote {args.out / 'neurons.jsonl'} and {args.out / 'controls.jsonl'}")
    if args.device in ("xpu", "cuda"):
        print("precision caveat: bf16 can move near-ties, so top lists are approximate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
